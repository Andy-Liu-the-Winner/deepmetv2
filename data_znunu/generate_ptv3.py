# System imports
import os
import glob
import os.path as osp

# External imports
import numpy as np
import time
from optparse import OptionParser
import torch
from tqdm import tqdm
from torch.utils.data import random_split
from torch_geometric.utils import is_undirected, to_undirected
from torch_geometric.data import (Data, Dataset)
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import concurrent.futures
from pathlib import Path

def conversion(rawpath, existing_pt_files, processed_dir):
    rawfile = rawpath.split('/')[-1]
    print("processing", rawfile)
    npzfile = np.load(rawpath,allow_pickle=True)
    print("file loaded")
    for ievt in range(np.shape(npzfile['x'])[1]):#file contains one event
        tic = time.time()
        print(rawfile, ievt)
        if rawfile.replace('.npz','_'+str(ievt)+'.pt') in existing_pt_files:
            print('already processed')
            continue
        else:
            print('Processing event', rawfile, ievt)
        inputs = np.array(npzfile['x'][:,ievt,:]).astype(np.float32)
        print(inputs.shape)
        #original: pt, eta, phi, d0, dz, mass, puppiWeight, pdgid, charge, frompv, pvref, pvAssocQuality
        inputs=inputs.T
        print(inputs.shape)
        #now: pX,pY,pT,eta,d0,dz,mass,puppiWeight,pdgId,charge,fromPV
        x = inputs[:,3:10]
        x=np.insert(x,0, inputs[:,0]*np.cos(inputs[:,2]),axis=1)
        x=np.insert(x,1, inputs[:,0]*np.sin(inputs[:,2]),axis=1)
        x=np.insert(x,2, inputs[:,0],axis=1)
        x=np.insert(x,3, inputs[:,1],axis=1)
        boolean = x[:,8]!=-999
        x=x[x[:,8]!=-999]
        x=x[x[:,9]!=-999]
        x = np.nan_to_num(x)
        x = np.clip(x, -5000., 5000.)
        assert not np.any(np.isnan(x))
        edge_index = torch.empty((2,0), dtype=torch.long)
        y = (np.array(npzfile['y'][ievt,:]).astype(np.float32)[None])
        #print(y)
        outdata = Data(x=torch.from_numpy(x),
                        edge_index=edge_index,
                        y=torch.from_numpy(y))
        print('saving...')
        try: torch.save(outdata, osp.join(processed_dir,(rawpath.replace('.npz','_'+str(ievt)+'.pt')).split('/')[-1] ))
        except: print('failed to save')
        print('saved')
        toc = time.time()
        print("time elapsed:", toc-tic)
    

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
    parser.add_option('-n', '--nworkers', help='number of workers', dest='nworkers', default=4)
    parser.add_option('-e', '--events', help='number of events', dest='nevents', default=None)
    (options, args) = parser.parse_args()
    dataset = options.dataset
    nevents = int(options.nevents)
    nworkers = int(options.nworkers)

    raw_paths = sorted(glob.glob(os.environ['PWD']+'/'+dataset+'/raw/*.npz')) #Finding all the raw files
    npz_files = [f.split('/')[-1] for f in raw_paths] #Getting the names of the raw files
    npz_files = [np.load(f,allow_pickle=True) for f in raw_paths] #Loading the raw files
    processed_dir = os.environ['PWD']+'/'+dataset+'/processed' #Setting the processed directory for saving pt files
    existing_pt_files = sorted(glob.glob(processed_dir+'/*_file*_slice_*_nevent_*.pt')) #Finding all the existing pt files
    existing_pt_files = [f.split('/')[-1] for f in existing_pt_files] #Getting the names of the existing pt files
    #print(raw_paths)
    for idx,raw_path in enumerate(tqdm(raw_paths)):
        raw_file = raw_path.split('/')[-1]
        npz_file = np.load(raw_path,allow_pickle=True)
        print(np.shape(npz_file['x']))
    if nevents is not None:
        events = 0
        raw_paths_temp = raw_paths
        raw_paths = []
        generator = np.random.default_rng()
        while events < nevents:
            index = generator.integers(0, len(raw_paths_temp), size=1)[0]
            raw_path = raw_paths_temp[index]
            raw_paths_temp.remove(raw_path)
            events += np.shape(np.load(raw_path,allow_pickle=True)['x'])[1]
            raw_paths.append(raw_path)
            print(np.shape(np.load(raw_path,allow_pickle=True)['x'])[1], events, len(raw_paths), index)
            
    with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
        futures = set()

        futures.update(executor.submit(conversion, rawpath, existing_pt_files, processed_dir) for rawpath in raw_paths)

        try:
            total = len(futures)
            processed = 0
            while len(futures) > 0:
                finished = set(job for job in futures if job.done())
                for job in finished:
                    processed += 1
                futures -= finished
            del finished
        except KeyboardInterrupt:
            print("Caught KeyboardInterrupt, terminating workers")
            for job in futures: job.cancel()
        except:
            for job in futures: job.cancel()
            raise




    
