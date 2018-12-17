import os, glob
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np
from torch.utils.data import Dataset
import torch
from torchvision import transforms
                    
#Display a tensor image
def DisplayTorch(tensor):
    tensor = tensor.squeeze().cpu()
    if len(tensor.size()) == 3:
        img = (tensor.permute(1,2,0).numpy() * 255).astype(np.uint8)
        plt.imshow(img)
    else:
        img = (tensor.numpy() * 255).astype(np.uint8)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))  
        
#Grayscale numpy image to sketch
def GrayToSketch(img):
    img_edge = cv2.GaussianBlur(img, (7,7), 0)
    img_edge = cv2.bitwise_not(cv2.Canny(img_edge, threshold1=20, threshold2=60))
    return img_edge
    
#RGB numpy image to sketch  
def RGBToSketch(img):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)   
    return GrayToSketch(img_grayscale)
                 
class MangaDataset(Dataset):
    blockSize = 45
    blurSize = 115
    numBlocks = 30
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = glob.glob(os.path.join(self.root_dir + '**/*.jpg'), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name_img = os.path.join(self.root_dir, self.files[idx])
        #Prepare the original color image
        image_pil = Image.open(name_img)
        if self.transform:
            image_pil = self.transform(image_pil)      
        image_numpy = np.array(image_pil)
        #Prepare the grayscale image        
        image_grayscale = cv2.cvtColor(image_numpy, cv2.COLOR_RGB2GRAY)   
        #Prepare the edge image
        image_edge = GrayToSketch(image_grayscale)
        #Whiten random regions of the image(as dropout) and then blur it
        rand_blocks = np.random.randint(low=0, high=image_numpy.shape[0] - self.blockSize, size=(self.numBlocks, 2))
        for i in range(self.numBlocks):
            image_numpy[rand_blocks[i][0] : rand_blocks[i][0] + self.blockSize, rand_blocks[i][1] :  rand_blocks[i][1] + self.blockSize, :] = 255
        image_hint = cv2.GaussianBlur(image_numpy, (self.blurSize, self.blurSize), 0)
        #Convert to tensors
        tensor_original = transforms.ToTensor()(image_pil)
        tensor_grayscale = torch.from_numpy(image_grayscale).unsqueeze(0).float() / 255
        tensor_sketch = torch.from_numpy(image_edge).unsqueeze(0).float() / 255
        tensor_hint = torch.from_numpy(image_hint).permute(2, 0, 1).float() / 255
        return {'original':tensor_original, 'grayscale':tensor_grayscale, 'sketch':tensor_sketch, 'hint':tensor_hint, 'filename':name_img}


class DraftDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.files = glob.glob(os.path.join(self.root_dir + '**/*_sketch.png'), recursive=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_sketch = os.path.join(self.root_dir, self.files[idx])                 
        with Image.open(img_sketch.replace('sketch', 'original')) as image:
            tensor_original = transforms.ToTensor()(image)
        with Image.open(img_sketch) as image:
            tensor_sketch = transforms.ToTensor()(image)[0,:,:].unsqueeze(0)
        with Image.open(img_sketch.replace('sketch', 'hint')) as image:
            tensor_hint = transforms.ToTensor()(image)
        with Image.open(img_sketch.replace('sketch', 'draft')) as image:
            tensor_draft = transforms.ToTensor()(image)   
        return {'original':tensor_original, 'sketch':tensor_sketch, 'hint':tensor_hint, 'draft':tensor_draft}

        
def LoadModel(model, path):
    model.load_state_dict(torch.load(path))
    
def SaveModel(model, path):
    torch.save(model.state_dict(), path)