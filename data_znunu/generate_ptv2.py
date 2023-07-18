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

class METDataset(Dataset):
    """PyTorch geometric dataset from processed hit information"""
    
    def __init__(self, root): 
        super(METDataset, self).__init__(root)
    
    def download(self):
        pass #download from xrootd or something later
    
    @property
    def raw_file_names(self):
        if not hasattr(self,'input_files'):
            #self.input_files = sorted(glob.glob(self.raw_dir+'/tt_file0_*.npz'))
            self.input_files = sorted(glob.glob(self.raw_dir+'/*.npz'))
        return [f.split('/')[-1] for f in self.input_files]

    @property
    def existing_pt_names(self):
        if not hasattr(self,'pt_files'):
            self.pt_files = sorted(glob.glob(self.processed_dir+'/*_file*_slice_*_nevent_*.pt')) #adjusted for current naming scheme
        return [f.split('/')[-1] for f in self.pt_files]
    
    @property
    def processed_file_names(self):
        if not hasattr(self,'processed_files'):
            #print(self.raw_file_names)
            #FIXME need to figure out how to get a list of expected pt files
            proc_names = [idx for idx in self.existing_pt_names]
            #proc_names = ['data_{}.pt'.format(idx) for idx in range(len(self.raw_file_names))]
            self.processed_files = [osp.join(self.processed_dir,name) for name in proc_names]
        return self.processed_files
    
    def __len__(self):
        return len(self.processed_file_names)
   
    def len(self):
        return len(self.processed_file_names)
 
    def get(self, idx):
        data = torch.load(self.processed_files[idx])
        return data
    
    def process(self):
        print('processing raw data...')
        #convert the npz into pytorch tensors and save them
        path = self.processed_dir
        npzfiles = []
        for idx,raw_path in enumerate(tqdm(self.raw_paths)):
            # print('raw path:', raw_path)
            # print('existing processed:', self.existing_pt_names)
            rawfile = raw_path.split('/')[-1]
            # print('raw file:', rawfile)
            npzfile = np.load(raw_path,allow_pickle=True)
            npzfiles.append(npzfile)

        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
            for ievt in range(np.shape(npzfile['x'])[1]):#file contains one event
                print(rawfile, ievt)
                if rawfile.replace('.npz','_'+str(ievt)+'.pt') in self.existing_pt_names:
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
                y = (np.array(npzfile['y'][:]).astype(np.float32)[None])
                #print(y)
                outdata = Data(x=torch.from_numpy(x),
                                edge_index=edge_index,
                                y=torch.from_numpy(y))
                print('saving...')
                torch.save(outdata, osp.join(self.processed_dir,(raw_path.replace('.npz','_'+str(ievt)+'.pt')).split('/')[-1] ))

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
        y = (np.array(npzfile['y'][:]).astype(np.float32)[None])
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
    (options, args) = parser.parse_args()
    dataset = options.dataset
    nworkers = 4

    raw_paths = sorted(glob.glob(os.environ['PWD']+'/'+dataset+'/raw/*.npz'))
    npz_files = [f.split('/')[-1] for f in raw_paths]
    npz_files = [np.load(f,allow_pickle=True) for f in raw_paths]
    processed_dir = os.environ['PWD']+'/'+dataset+'/processed'
    existing_pt_files = sorted(glob.glob(processed_dir+'/*_file*_slice_*_nevent_*.pt'))
    existing_pt_files = [f.split('/')[-1] for f in existing_pt_files] 
    #print(raw_paths)
    for idx,raw_path in enumerate(tqdm(raw_paths)):
        raw_file = raw_path.split('/')[-1]
        npz_file = np.load(raw_path,allow_pickle=True)
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




    