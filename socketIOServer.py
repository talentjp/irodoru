#Socket io & Flask
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask, render_template
import numpy as np
#Torch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils, models
#Common libraries
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
from skimage import io, transform
from PIL import Image
import cv2
import scipy
#Project libraries
from stcutils import *
from stcmodels import *
from stcgan import *
from stcautocolor import *
from misc import *

def colorizeImg(img, downsample=False):
    #Compute the color            
    img_sketch, img_hint = PainterImageToSketchAndHint(img)        
    autoC.loadImages(img_sketch, img_hint, shrinkImages=downsample, edgeDetect=True)
    colored_tensor = autoC.getRefined().cpu()
    if downsample:
        alpha_tensor = torch.ones(1, 1, 256, 256)
    else:
        alpha_tensor = torch.ones(1, 1, 512, 512)
    colored_tensor = torch.cat([colored_tensor, alpha_tensor], dim=1)
    colored_numpy = (colored_tensor.squeeze().permute(1,2,0) * 255).byte().numpy()
    if downsample:
        colored_numpy = cv2.resize(colored_numpy, (512,512) , interpolation=cv2.INTER_CUBIC)            
    return colored_numpy #32-bit RGBA image


autoC = AutoColor(enableCUDA=True)
#Replace this with your model path for deployment
autoC.loadModels(path_draft='model_epoch_0001_batch_00035999.pt', 
                 path_refine='refinement_model_epoch_0010_batch_00003999.pt')
                
sio = socketio.Server()
app = Flask(__name__)

@app.route('/')
def index():
    """Serve the client-side application."""
    return render_template('index.html')

@sio.on('connect', namespace='/colorization')
def connect(sid, environ):    
    print("connected: ", sid)
    
@sio.on('image', namespace='/colorization')
def receiveImage(sid, data):
    img = np.frombuffer(data['buffer'], dtype=np.uint8) 
    img = img.reshape(512,512,4,order='C')
    colored_numpy = colorizeImg(img, downsample=False)    
    sio.emit('coloring', colored_numpy.tobytes(), namespace='/colorization')  

@sio.on('disconnect', namespace='/colorization')
def disconnect(sid):
    print('disconnected: ', sid)

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)