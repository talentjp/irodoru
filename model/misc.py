import os, glob
import numpy as np
import scipy.ndimage.filters as fi
import matplotlib.pyplot as plt
from model.stcutils import *
import torch
from scipy.ndimage.morphology import binary_dilation
import requests

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

def gkern2(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    # create nxn zeros
    inp = np.zeros((kernlen, kernlen))
    # set element at the middle to one, a dirac delta
    inp[kernlen//2, kernlen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    return fi.gaussian_filter(inp, nsig)

def randomSpray(img, lower_bound = 15, upper_bound = 210, density = 40, n_lines = 4):
    kernel_3d = np.zeros((21,21,3))
    kernel_3d[:,:,0] = gkern2() * 3
    kernel_3d[:,:,1] = gkern2() * 3
    kernel_3d[:,:,2] = gkern2() * 3
    for i in range(n_lines):
        startX, startY = np.random.randint(lower_bound, upper_bound, size=2)
        color = img[startX, startY, :]    
        endX, endY = np.random.randint(lower_bound, upper_bound, size=2)
        img = img.astype('float32')
        for i in range(100):
            mean = [0, 0]
            cov = [[1,0], [0,1]]
            x, y = np.random.multivariate_normal(mean, cov, density).T
            x = (x * 20).astype(np.int32) + startX + int(i * (endX - startX) / 100)
            y = (y * 20).astype(np.int32) + startY + int(i * (endY - startY) / 100)
            for k_i in range(density):
                #Make sure it's within range
                if x[k_i] > lower_bound and x[k_i] < upper_bound and y[k_i] > lower_bound and y[k_i] < upper_bound:
                    img[x[k_i] - 10 : x[k_i] + 11, y[k_i] - 10: y[k_i] + 11, :] = kernel_3d * color + (1 - kernel_3d) * img[x[k_i] - 10 : x[k_i] + 11, y[k_i] - 10: y[k_i] + 11, :]
                
    img[img > 255] = 255
    img = img.astype('uint8')
    return img  
    
#Usage : for i, data in enumerate(log_progress(gan_draft.train_dataset)):   
def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )   

#Delete all non-rgb(e.g. grayscale) images
def RemoveNonRGBImages(root_folder):
    for f in glob.glob(os.path.join(root_folder + '**/*.jpg'), recursive=True):
        shouldDelete = False
        with Image.open(f) as image:
            if image.mode != 'RGB':
                shouldDelete = True
        if shouldDelete:
            os.remove(f)
                        
#Plot an item in the dataset
def PlotData(data, model):
    model.eval()
    plt.figure(figsize=(50,50))
    plt.subplot(1,3,1)
    DisplayTorch(data['original'])
    plt.subplot(1,3,2)
    DisplayTorch(data['sketch'])
    with torch.no_grad():
        result = model(torch.cat([data['sketch'].unsqueeze(0).cuda(), data['hint'].unsqueeze(0).cuda()], dim=1))    
    plt.subplot(1,3,3)    
    DisplayTorch(result)   
    
#Compare models on the same data    
def CompareModels(data, model, model_path1, model_path2 = None):
    plt.figure(figsize=(50,50))
    plt.subplot(1,3,1)
    DisplayTorch(data['original'])
    model.loadModels(model_path1)
    model.G.eval()    
    sketch = data['sketch'].unsqueeze(0)
    hint = data['hint'].unsqueeze(0)    
    if model.cuda_enabled:
        sketch = sketch.cuda()
        hint = hint.cuda()          
    with torch.no_grad():       
        result = model.G(torch.cat([sketch, hint], dim=1))        
    plt.subplot(1,3,2)
    DisplayTorch(result)
    if model_path2 is not None: 
        model.loadModels(model_path2)
        model.G.eval()    
        sketch = data['sketch'].unsqueeze(0)
        hint = data['hint'].unsqueeze(0)    
        if model.cuda_enabled:
            sketch = sketch.cuda()
            hint = hint.cuda()                  
        with torch.no_grad():
            result = model.G(torch.cat([sketch, hint], dim=1)) 
        plt.subplot(1,3,3)
        DisplayTorch(result)            
        
def CompareDraftRefined(gan_draft, gan_refinement, data, saveToFile=False):
    fig = plt.figure(figsize=(50,50))
    plt.subplot(1,3,1)
    DisplayTorch(data['original'])    
    gan_draft.G.eval()    
    with torch.no_grad():
        result = gan_draft.G(torch.cat([data['sketch'].unsqueeze(0).cuda(), data['hint'].unsqueeze(0).cuda()], dim=1))        
    plt.subplot(1,3,2)
    DisplayTorch(result)
    gan_refinement.G.eval()    
    with torch.no_grad():
        result = gan_refinement.G(torch.cat([data['sketch'].unsqueeze(0).cuda(), data['hint'].unsqueeze(0).cuda(), result.cuda()], dim=1))  
    plt.subplot(1,3,3)
    DisplayTorch(result) 
    if saveToFile:
        fig.savefig('plot_{}.png'.format(os.path.basename(data['filename']).replace('.jpg', '')))       
        plt.close() 
        
def PainterImageToSketchAndHint(img):
    img_sketch = img.copy()[:,:,:3]
    img_hint = img.copy()[:,:,:3]
    #grayscale images have A and B channels at 0, we give it a tolerance of +-2
    img_lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)  
    mask = (np.abs(img_lab[:,:,1] - 128) > 2) | (np.abs(img_lab[:,:,2] - 128) > 2)
    #Mask out the color
    img_sketch[:,:,0][mask] = 255
    img_sketch[:,:,1][mask] = 255
    img_sketch[:,:,2][mask] = 255
    #Mask out the ~color
    img_hint[:,:,0][~mask] = 255
    img_hint[:,:,1][~mask] = 255
    img_hint[:,:,2][~mask] = 255      
    return img_sketch, img_hint