import argparse, sys, os
from model.misc import *
from model.stcgan import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='tools for training the models')
    parser.add_argument('-C', '--clean', 
                        help='Path to the danbooru dataset that needs cleaning') 
    parser.add_argument('-D', '--draft', nargs='*',
                        help='Training draft model. usage: --draft <path to danbooru dataset> <path to save the models> [<number of epochs(default:100)>]') 
    parser.add_argument('-R', '--refine', nargs='*',
                        help='Training refinement model. usage: --draft <path to draft dataset> <path to save the models> [<number of epochs(default:100)>]') 
    parser.add_argument('-M', '--model', 
                        help='path to previously saved model')
    parser.add_argument('-G', '--generate', nargs=2, 
                        help='Generating the draft images. usage: --generate <path to danbooru dataset> <path to generated images>')    
    results = parser.parse_args(sys.argv[1:])

    if results.clean is not None:
        RemoveNonRGBImages(results.clean)
    elif results.draft is not None:    
        if len(results.draft) < 2:
            print('Missing path(s). usage: --draft <path to danbooru dataset> <path to save the models> [<number of epochs(default:100)>]')
        else:
            gan = DraftGAN(results.draft[0], results.draft[1])
            if results.model is not None:
                gan.loadModels(results.model)
            num_epochs = 100
            if len(results.draft) == 3:
                num_epochs = int(results.draft[2])
            gan.train(num_epochs)
    elif results.refine is not None:
        if len(results.refine) < 2:
            print('Missing path(s). usage: --draft <path to draft dataset> <path to save the models> [<number of epochs(default:100)>]')
        else:
            gan = RefineGAN(results.refine[1], results.refine[0])
            if results.model is not None:
                gan.loadModels(results.model)            
            num_epochs = 100
            if len(results.refine) == 3:
                num_epochs = int(results.refine[2])
            gan.train(num_epochs)
    elif results.generate is not None:
        if results.model is None:
            print('Please specify the model using --model')            
        else:
            gan = DraftGAN(results.generate[0])
            gan.loadModels(results.model)
            gan.saveDraftImages(results.generate[1])