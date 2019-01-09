from server import socketIOServer
from server import staticServer
import argparse, sys, os
from multiprocessing import Process, set_start_method
from model.misc import *
from model.stcautocolor import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='launcher for all services')
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

    set_start_method('spawn')
    p1 = Process(target=socketIOServer.startServer, args=(results.model_draft, results.model_refine))
    p2 = Process(target=staticServer.startServer)
    p1.start()
    p2.start()

