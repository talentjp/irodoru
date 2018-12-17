import os
from PIL import Image
import cv2
import numpy as np
import torch
from stcgan import *
from stcutils import *

class AutoColor:
    def __init__(self, enableCUDA = True):
        self.cuda_enabled = torch.cuda.is_available() and enableCUDA
        self.gan_draft = DraftGAN(enableCUDA = self.cuda_enabled)
        self.gan_refine = RefineGAN(enableCUDA = self.cuda_enabled)
        self.tensor_draft = None
        self.tensor_sketch = None
        self.tensor_hint = None
        self.tensor_refined = None
        self.count = 0
        
    def loadModels(self, path_draft, path_refine):
        self.gan_draft.loadModels(path_draft)
        self.gan_refine.loadModels(path_refine)
        
    def loadImages(self, img_sketch, img_hint, shrinkImages = False, edgeDetect = False):
        if type(img_sketch) == str: 
            with Image.open(img_sketch) as img:
                self.numpy_sketch = np.array(img)[:,:,:3]
        else:
            self.numpy_sketch = img_sketch[:,:,:3]
        if type(img_hint) == str:
            with Image.open(img_hint) as img:
                self.numpy_hint = np.array(img)[:,:,:3] #Make sure it's 3 channels
        else:
            self.numpy_hint = img_hint[:,:,:3]
        self.numpy_sketch = cv2.cvtColor(self.numpy_sketch, cv2.COLOR_RGB2GRAY)     
        if shrinkImages:
            self.numpy_sketch = cv2.resize(self.numpy_sketch, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
            self.numpy_hint    = cv2.resize(self.numpy_hint, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
        if edgeDetect:
            self.numpy_sketch = GrayToSketch(self.numpy_sketch)
        self.numpy_hint = cv2.GaussianBlur(self.numpy_hint, (115, 115), 0)
        self.tensor_draft = None
        self.tensor_sketch = None
        self.tensor_hint = None
        self.tensor_refined = None      
        
    def getDraft(self):
        if self.tensor_draft is None:                                       
            self.tensor_sketch = torch.from_numpy(self.numpy_sketch).unsqueeze(0).unsqueeze(0).float() / 255
            self.tensor_hint    = torch.from_numpy(self.numpy_hint).permute(2, 0, 1).unsqueeze(0).float() / 255          
            if self.cuda_enabled:
                self.tensor_sketch = self.tensor_sketch.cuda()
                self.tensor_hint = self.tensor_hint.cuda()          
                self.gan_draft.G = self.gan_draft.G.cuda()
            self.gan_draft.G.eval()         
            with torch.no_grad():
                self.tensor_draft  = self.gan_draft.G(torch.cat([self.tensor_sketch, self.tensor_hint], dim=1))
            
        return self.tensor_draft
        
    def getRefined(self):
        if self.tensor_refined is None:                 
            if self.tensor_draft is not None:
                if self.cuda_enabled:
                    self.gan_refine.G = self.gan_refine.G.cuda()
                self.gan_refine.G.eval()
                with torch.no_grad():
                    self.tensor_refined  = self.gan_refine.G(torch.cat([self.tensor_sketch, self.tensor_hint, self.tensor_draft], dim=1))
            else:
                self.getDraft()
                self.getRefined()
                
        return self.tensor_refined       

    def plotImages(self, saveToFile = False):
        fig = plt.figure(figsize=(50,50))
        ax = plt.subplot(1,4,1)
        ax.set_title('Sketch', fontsize=50)
        DisplayTorch(self.tensor_sketch)       
        ax = plt.subplot(1,4,2)
        ax.set_title('Color Hint', fontsize=50)
        DisplayTorch(self.tensor_hint)
        ax = plt.subplot(1,4,3)
        ax.set_title('Draft', fontsize=50)
        DisplayTorch(self.tensor_draft) 
        ax = plt.subplot(1,4,4)
        ax.set_title('Refined', fontsize=50)
        DisplayTorch(self.tensor_refined)                   
        if saveToFile:
            fig.savefig('Plot{}.png'.format(self.count))
            plt.close() 
            self.count += 1