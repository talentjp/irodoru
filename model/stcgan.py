import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import utils
from model.stcmodels import *
from model.stcutils import *
from unet.unet_model import *
import os

class DraftGAN:
    def __init__(self, train_path = None, model_path = None, enableCUDA = True):
        self.train_dataset = None
        self.train_loader = None
        self.train_path = train_path
        self.model_path = model_path
        if self.train_path is not None:
            self.train_path = os.path.join(self.train_path, '')
        if self.model_path is not None:
            self.model_path = os.path.join(self.model_path, '')            
        self.epoch = 0
        self.train_history = []  #A list of past dictionaries(g loss, d loss...etc.)
        self.cuda_enabled = torch.cuda.is_available() and enableCUDA
        self.G = UNet(4, 3)
        self.D = Discriminator()
        self.MSE = nn.MSELoss()
        self.L1 = nn.L1Loss()
        self.BCE = nn.BCELoss()
        self.G_optimizer = Adam(self.G.parameters(), lr=1e-2, weight_decay=0.0)
        self.D_optimizer = Adam(self.D.parameters(), lr=1e-2, weight_decay=0.0)
        if self.train_path is not None:
            self.train_dataset = MangaDataset(root_dir=self.train_path, transform=transforms.Compose([
                                                transforms.Resize(512),
                                                transforms.RandomCrop(256)
                                            ]))   
        if self.train_dataset is not None:                                             
            self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        if self.cuda_enabled:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.L1 = self.L1.cuda()
            self.BCE = self.BCE.cuda()
                                
    def train(self, num_epochs):
        #don't even allow training if there is no data loader
        if self.train_loader is None:
            return
        self.G.train()
        self.D.train()
        for self.epoch in range(self.epoch, self.epoch + num_epochs):
            total_epoch_G_loss = 0
            total_epoch_D_loss = 0          
            for batch_idx, img_dict in enumerate(self.train_loader):
                if self.cuda_enabled:
                    real_images = img_dict['original'].cuda()
                    sketch_images = img_dict['sketch'].cuda()
                    hint_images   = img_dict['hint'].cuda()
                else:
                    real_images = img_dict['original']
                    sketch_images = img_dict['sketch']
                    hint_images   = img_dict['hint']            
                
                ##############   Generator Step   ##################
                self.G_optimizer.zero_grad()
                #Sketch + hint for G
                G_input_data = torch.cat([sketch_images, hint_images], dim=1)                        
                fake_images = self.G(G_input_data)
                D_fake_input_data = torch.cat([fake_images, hint_images], dim=1)
                prediction_from_fake = self.D(D_fake_input_data)        
                #Generator wants Discriminator to mark all images it generates as real and visually close to the real
                G_loss = 0.01 * self.BCE(prediction_from_fake, torch.ones_like(prediction_from_fake)) + self.L1(fake_images, real_images)           
                #This makes the network harder to train but boosts the saturation
                #mean_image = fake_images.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).expand_as(fake_images).detach()
                #G_loss = 0.01 * self.BCE(prediction_from_fake, torch.ones_like(prediction_from_fake)) + self.L1(fake_images, real_images) - 0.1 * self.MSE(fake_images, mean_image)
                #Gradient Descent on G                        
                G_loss.backward()
                self.G_optimizer.step()
                
                ##############   Discriminator Step   ##################
                self.D_optimizer.zero_grad()                                        
                fake_images = self.G(G_input_data)
                #Real + hint
                D_real_input_data = torch.cat([real_images, hint_images], dim=1)
                prediction_from_real = self.D(D_real_input_data)        
                #Fake + hint
                D_fake_input_data = torch.cat([fake_images, hint_images], dim=1)
                prediction_from_fake = self.D(D_fake_input_data)
                #Discriminator wants to mark all generator images as fake, all real images as real            
                 #Spray some noise (label flipping) to give D a hard time (so G has a chance to fight against it)
                D_loss = (self.BCE(prediction_from_real, torch.ones_like(prediction_from_real) - (torch.rand_like(prediction_from_real) < 0.05).float()) +
                              self.BCE(prediction_from_fake, (torch.rand_like(prediction_from_real) < 0.05).float()))                                  
                             
                #Gradient Descent on D
                D_loss.backward()
                self.D_optimizer.step()    
                
                total_epoch_G_loss += G_loss
                total_epoch_D_loss += D_loss                
                #Print useful information every 200 batches
                if batch_idx % 200 == 0:
                    print('Epoch : {}, Batch : {}, G loss : {}, D loss : {}'.format(self.epoch, batch_idx, G_loss, D_loss))
                if batch_idx % 1000 == 999: #Save a checkpoint every 1000 batches
                    avg_batch_G_loss = total_epoch_G_loss / (batch_idx + 1)
                    avg_batch_D_loss = total_epoch_D_loss / (batch_idx + 1)
                    self.train_history.append({'g_loss':avg_batch_G_loss, 'd_loss':avg_batch_D_loss})               
                    self.saveModels(os.path.join(self.model_path, 'draft_model_epoch_{:04d}_batch_{:08d}.pt'.format(self.epoch, batch_idx)))
            #Save a model for each epoch
            self.saveModels(os.path.join(self.model_path, 'draft_model_epoch_{:04d}.pt'.format(self.epoch)))
                        
    def loadModels(self, path):
        checkpoint = torch.load(path)
        #self.G.load_state_dict(checkpoint['G_state_dict'])
        #to make PyTorch 0.4.0 load 0.4.1 models
        pretrained_dict = checkpoint['G_state_dict']
        model_dict = self.G.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}     
        self.G.load_state_dict(pretrained_dict)         
        #self.D.load_state_dict(checkpoint['D_state_dict'])
        pretrained_dict = checkpoint['D_state_dict']
        model_dict = self.D.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}     
        self.D.load_state_dict(pretrained_dict)             
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.train_history = checkpoint['history']
        
    def saveModels(self, path):
        if model.model_path is None:
            return
        checkpoint = {'epoch':self.epoch, 'G_state_dict':self.G.state_dict(), 'D_state_dict':self.D.state_dict(), 
                           'G_optimizer_state_dict':self.G_optimizer.state_dict(), 'D_optimizer_state_dict':self.D_optimizer.state_dict(),
                           'history':self.train_history}
        torch.save(checkpoint, path)    
        
    def getLosses(self):
        g_losses = []
        d_losses = []
        for history_item in self.train_history:
            g_losses.append(history_item['g_loss'].item())
            d_losses.append(history_item['d_loss'].item())
        return g_losses, d_losses

    def saveDraftImages(self, path):
        path = os.path.join(path, '')
        if self.cuda_enabled:
            self.G = self.G.cuda()
        self.G.eval()
        with torch.no_grad():
            for i in range(len(self.train_dataset)):
                data = self.train_dataset[i]    
                #for i, data in enumerate(log_progress(gan_draft.train_dataset)):
                output = self.G(torch.cat([data['sketch'].unsqueeze(0).cuda(), data['hint'].unsqueeze(0).cuda()], dim=1))
                output = output.squeeze()
                #Save the original and the generated version
                utils.save_image(data['original'], os.path.join(path, '{:08d}_original.png'.format(i)))
                utils.save_image(data['hint'], os.path.join(path, '{:08d}_hint.png'.format(i)))
                utils.save_image(data['sketch'], os.path.join(path, '{:08d}_sketch.png'.format(i)))
                utils.save_image(output, os.path.join(path, '{:08d}_draft.png'.format(i)))         
        
class RefineGAN:
    def __init__(self, model_path = None, draft_path = None, enableCUDA = True):
        self.train_dataset = None
        self.train_loader = None
        self.model_path = model_path
        self.draft_path = draft_path  
        if self.draft_path is not None:
            self.draft_path = os.path.join(self.draft_path, '')
        if self.model_path is not None:
            self.model_path = os.path.join(self.model_path, '')         
        self.epoch = 0
        self.train_history = []  #A list of dictionaries(g loss, d loss...etc.)
        self.cuda_enabled = torch.cuda.is_available() and enableCUDA
        self.G = UNet(7, 3)   #Sketch(1) + Hint(3) + Draft(3)
        self.D = Discriminator()
        self.L1 = nn.L1Loss()
        self.BCE = nn.BCELoss() #Binary Cross Entropy
        self.G_optimizer = Adam(self.G.parameters(), lr=1e-2, weight_decay=0.0)
        self.D_optimizer = Adam(self.D.parameters(), lr=1e-2, weight_decay=0.0)
        if self.draft_path is not None:
            self.train_dataset = DraftDataset(root_dir=self.draft_path)                                                
        if self.train_dataset is not None:
            self.train_loader = DataLoader(self.train_dataset, batch_size=8, shuffle=True)
        #Enable CUDA if possible
        if self.cuda_enabled:
            self.G = self.G.cuda()
            self.D = self.D.cuda()
            self.L1 = self.L1.cuda()
            self.BCE = self.BCE.cuda()
                                
    def train(self, num_epochs):
        if self.train_loader is None:
            return
        self.G.train()
        self.D.train()
        for self.epoch in range(self.epoch, self.epoch + num_epochs):
            total_epoch_G_loss = 0
            total_epoch_D_loss = 0          
            for batch_idx, img_dict in enumerate(self.train_loader):
                if self.cuda_enabled:
                    real_images = img_dict['original'].cuda()
                    sketch_images = img_dict['sketch'].cuda()
                    hint_images   = img_dict['hint'].cuda()
                    draft_images = img_dict['draft'].cuda()
                else:
                    real_images = img_dict['original']
                    sketch_images = img_dict['sketch']
                    hint_images   = img_dict['hint']
                    draft_images = img_dict['draft']                    
                
                ##############   Generator Step   ##################
                self.G_optimizer.zero_grad()
                #Sketch + hint for G
                G_input_data = torch.cat([sketch_images, hint_images, draft_images], dim=1)                        
                fake_images = self.G(G_input_data)
                D_fake_input_data = torch.cat([fake_images, hint_images], dim=1)
                prediction_from_fake = self.D(D_fake_input_data)        
                #Generator wants Discriminator to mark all images it generates as real and visually close to the real
                G_loss = 0.01 * self.BCE(prediction_from_fake, torch.ones_like(prediction_from_fake)) + self.L1(fake_images, real_images)           
                #This makes the network harder to train (decrease the weight?)
                #mean_image = fake_images.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).expand_as(fake_images).detach()
                #G_loss = 0.01 * self.BCE(prediction_from_fake, torch.ones_like(prediction_from_fake)) + self.L1(fake_images, real_images) - 0.1 * self.MSE(fake_images, mean_image)
                #Gradient Descent on G                        
                G_loss.backward()
                self.G_optimizer.step()
                
                ##############   Discriminator Step   ##################
                self.D_optimizer.zero_grad()                                        
                fake_images = self.G(G_input_data)
                #Real + hint
                D_real_input_data = torch.cat([real_images, hint_images], dim=1)
                prediction_from_real = self.D(D_real_input_data)        
                #Fake + hint
                D_fake_input_data = torch.cat([fake_images, hint_images], dim=1)
                prediction_from_fake = self.D(D_fake_input_data)
                #Discriminator wants to mark all generator images as fake, all real images as real                
                 #Spray some noise (label flipping) to give D a hard time (so G has a chance to fight against it)
                D_loss = (self.BCE(prediction_from_real, torch.ones_like(prediction_from_real) - (torch.rand_like(prediction_from_real) < 0.05).float()) +
                              self.BCE(prediction_from_fake, (torch.rand_like(prediction_from_real) < 0.05).float()))                            
                             
                #Gradient Descent on D
                D_loss.backward()
                self.D_optimizer.step()    
                
                total_epoch_G_loss += G_loss
                total_epoch_D_loss += D_loss                
                #Print useful information every 100 batches
                if batch_idx % 100 == 0:
                    print('Epoch : {}, Batch : {}, G loss : {}, D loss : {}'.format(self.epoch, batch_idx, G_loss, D_loss))
                if batch_idx % 1000 == 999:
                    avg_batch_G_loss = total_epoch_G_loss / (batch_idx + 1)
                    avg_batch_D_loss = total_epoch_D_loss / (batch_idx + 1)
                    self.train_history.append({'g_loss':avg_batch_G_loss, 'd_loss':avg_batch_D_loss})               
                    self.saveModels(os.path.join(self.model_path, 'refinement_model_epoch_{:04d}_batch_{:08d}.pt'.format(self.epoch, batch_idx)))
            #Save a model for each epoch
            print("saving model to " + os.path.join(self.model_path, 'refinement_model_epoch_{:04d}.pt'.format(self.epoch)))
            self.saveModels(os.path.join(self.model_path, 'refinement_model_epoch_{:04d}.pt'.format(self.epoch)))                    
        
    def loadModels(self, path):
        checkpoint = torch.load(path)
        self.G.load_state_dict(checkpoint['G_state_dict'])
        self.D.load_state_dict(checkpoint['D_state_dict'])
        self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
        self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])
        self.epoch = checkpoint['epoch'] + 1
        self.train_history = checkpoint['history']
        
    def saveModels(self, path):
        if self.model_path is None:
            return
        checkpoint = {'epoch':self.epoch, 'G_state_dict':self.G.state_dict(), 'D_state_dict':self.D.state_dict(), 
                           'G_optimizer_state_dict':self.G_optimizer.state_dict(), 'D_optimizer_state_dict':self.D_optimizer.state_dict(),
                           'history':self.train_history}
        torch.save(checkpoint, path)    
        
    def getLosses(self):
        g_losses = []
        d_losses = []
        for history_item in self.train_history:
            g_losses.append(history_item['g_loss'].item())
            d_losses.append(history_item['d_loss'].item())
        return g_losses, d_losses               