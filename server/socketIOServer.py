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
from model.stcutils import *
from model.stcmodels import *
from model.stcgan import *
from model.stcautocolor import *
from model.misc import *

autoC = AutoColor(enableCUDA=True)

def colorizeImg(img, downsample=False):
    global autoC
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


def startServer(model_draft, model_refine):    
    global autoC
    global app
    global sio    
    #Replace this with your model path for deployment
    autoC.loadModels(path_draft=model_draft, path_refine=model_refine)                    
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)
    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 8000)), app)    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='inference service')
    parser.add_argument('-Md', '--model_draft', 
                        help='path to draft model')
    parser.add_argument('-Mr', '--model_refine', 
                        help='path to refinement model')    
    results = parser.parse_args(sys.argv[1:])
    if results.model_draft is None:
        if not os.path.isfile('default_draft.pt'):
            print('default draft model does not exist, downloading from Google Drive......')
            download_file_from_google_drive('1vRJH4hywJI02Ta7e2XQlt7u21jIUMqh_', 'default_draft.pt')
        results.model_draft = 'default_draft.pt'
    if results.model_refine is None:
        if not os.path.isfile('default_refine.pt'):
            print('default refinement model does not exist, downloading from Google Drive......')
            download_file_from_google_drive('1jWUdo3k-gbx8N6dj1QiN0EDcCagH7Lhy', 'default_refine.pt')
        results.model_refine = 'default_refine.pt'
    startServer(results.model_draft, results.model_refine)