from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import NanoAODSchema,BaseSchema
import numpy as np
import os
from optparse import OptionParser
import glob
import awkward as ak
import time
import json
from collections import OrderedDict,defaultdict
import concurrent.futures
from pathlib import Path
recdd = lambda : defaultdict(recdd) ## define recursive defaultdict
JSON_LOC = 'filelist.json'

def multidict_tojson(filepath, indict):
    ## expand into multidimensional dictionary
    with open(filepath, "w") as fo:
        json.dump( indict, fo)
        print("save to %s" %filepath)

def future_savez(i, events_slice, nparticles_per_event):
        tic=time.time()
        genmet_list = np.column_stack([
                events_slice.GenMET.pt * np.cos(events_slice.GenMET.phi),
                events_slice.GenMET.pt * np.sin(events_slice.GenMET.phi),
                events_slice.MET.pt * np.cos(events_slice.MET.phi),
                events_slice.MET.pt * np.sin(events_slice.MET.phi),
                events_slice.PuppiMET.pt * np.cos(events_slice.PuppiMET.phi),
                events_slice.PuppiMET.pt * np.sin(events_slice.PuppiMET.phi),
                events_slice.DeepMETResponseTune.pt * np.cos(events_slice.DeepMETResponseTune.phi),
                events_slice.DeepMETResponseTune.pt * np.sin(events_slice.DeepMETResponseTune.phi),
                events_slice.DeepMETResolutionTune.pt * np.cos(events_slice.DeepMETResolutionTune.phi),
                events_slice.DeepMETResolutionTune.pt * np.sin(events_slice.DeepMETResolutionTune.phi),
                events_slice.LHE.HT
        ])
        genmet_list = np.array(genmet_list)

        particle_list = np.full((12,len(events_slice),nparticles_per_event),-999, dtype='float32')
        particle_list[0] = ak.fill_none(ak.pad_none(events_slice.PFCands.pt, nparticles_per_event,clip=True),-999)
        particle_list[1] = ak.fill_none(ak.pad_none(events_slice.PFCands.eta, nparticles_per_event,clip=True),-999)          
        particle_list[2] = ak.fill_none(ak.pad_none(events_slice.PFCands.phi, nparticles_per_event,clip=True),-999)          
        particle_list[3] = ak.fill_none(ak.pad_none(events_slice.PFCands.d0, nparticles_per_event,clip=True),-999)           
        particle_list[4] = ak.fill_none(ak.pad_none(events_slice.PFCands.dz, nparticles_per_event,clip=True),-999)           
        particle_list[5] = ak.fill_none(ak.pad_none(events_slice.PFCands.mass, nparticles_per_event,clip=True),-999)         
        particle_list[6] = ak.fill_none(ak.pad_none(events_slice.PFCands.puppiWeight, nparticles_per_event,clip=True),-999) 
        particle_list[7] = ak.fill_none(ak.pad_none(events_slice.PFCands.pdgId, nparticles_per_event,clip=True),-999)        
        particle_list[8] = ak.fill_none(ak.pad_none(events_slice.PFCands.charge, nparticles_per_event,clip=True),-999)        
        particle_list[9] = ak.fill_none(ak.pad_none(events_slice.PFCands.fromPV, nparticles_per_event,clip=True),-999) 
              
        particle_list[10] = ak.fill_none(ak.pad_none(events_slice.PFCands.pvRef, nparticles_per_event,clip=True),-999)         
        particle_list[11] = ak.fill_none(ak.pad_none(events_slice.PFCands.pvAssocQuality, nparticles_per_event,clip=True),-999)

        # particle_list = np.column_stack([
        #               events_slice.PFCands.pt[i],
        #               events_slice.PFCands.eta[i],
        #               events_slice.PFCands.phi[i],
        #               events_slice.PFCands.mass[i],
        #               events_slice.PFCands.d0[i],
        #               events_slice.PFCands.dz[i],
        #               events_slice.PFCands.pdgId[i],
        #               events_slice.PFCands.charge[i],
        #               events_slice.PFCands.fromPV[i],
        #               events_slice.PFCands.puppiWeight[i],
        #               events_slice.PFCands.pvRef[i],
        #               events_slice.PFCands.pvAssocQuality[i]
        # ])
        # particle_list = np.array(particle_list)

        # eventi = [particle_list,genmet_list]
        npz_file='/hildafs/projects/phy230010p/fep/DeepMETv2/data_znunu/'+dataset+'/raw/'+dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(events_slice))
        np.savez(npz_file,x=particle_list,y=genmet_list) 

        #toc=time.time()
        #print(toc-tic)
        # return eventi

def conversion(ifile):
    events = NanoEventsFactory.from_root(ifile, schemaclass=NanoAODSchema).events()
    nevents_total = len(events)
    print(ifile, ' Number of events:', nevents_total)
            
    for i in range(int(nevents_total / eventperfile)+1):
        if i< int(nevents_total / eventperfile):
            print('from ',i*eventperfile, ' to ', (i+1)*eventperfile)
            events_slice = events[i*eventperfile:(i+1)*eventperfile]
        elif i == int(nevents_total / eventperfile) and i*eventperfile<=nevents_total:
            print('from ',i*eventperfile, ' to ', nevents_total)
            events_slice = events[i*eventperfile:nevents_total]
        else:
            print(' weird ... ')

        nparticles_per_event = max(ak.num(events_slice.PFCands.pt, axis=1))
        print("max NPF in this range: ", nparticles_per_event)
        tic=time.time()
        future_savez(i, events_slice, nparticles_per_event) 
        toc=time.time()
        print('time:',toc-tic)

if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        parser.add_option('-s', '--startfile',type=int, default=0, help='startfile')
        parser.add_option('-e', '--endfile',type=int, default=1, help='endfile')
        (options, args) = parser.parse_args()
        datasetsname = {
            "znunu100to200": ['Znunu/ZJetsToNuNu_HT-100To200_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu200to400": ['Znunu/ZJetsToNuNu_HT-200To400_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu400to600": ['Znunu/ZJetsToNuNu_HT-400To600_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu600to800": ['Znunu/ZJetsToNuNu_HT-600To800_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu800to1200":[ 'Znunu/ZJetsToNuNu_HT-800To1200_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu1200to2500": ['Znunu/ZJetsToNuNu_HT-1200To2500_TuneCP5_13TeV-madgraphMLM-pythia8/'],
            "znunu2500toInf": ['Znunu/ZJetsToNuNu_HT-2500ToInf_TuneCP5_13TeV-madgraphMLM-pythia8/'],
        }

        # Be nice to eos, save list to a file
        #filelists = recdd()
        #for datset in datasetsname.keys():
        #    filelists[datset] = glob.glob('/eos/uscms/store/group/lpcjme/NanoMET/'+datasetsname[datset][0]+'/*/*/*/*root')
        #    filelists[datset] = [x.replace('/eos/uscms','root://cmseos.fnal.gov/') for x in filelists[datset] ]
        #multidict_tojson(JSON_LOC, filelists )
        #exit()
        dataset=options.dataset
        if dataset not in datasetsname.keys():
            print('choose one of them: ', datasetsname.keys())
            exit()
        #Read file from json
        with open(JSON_LOC, "r") as fo:
            file_names = json.load(fo)
        file_names = file_names[dataset]
        print('found ', len(file_names)," files")

        if options.startfile>=options.endfile and options.endfile!=-1:
            print("make sure options.startfile<options.endfile")
            exit()
        file_names = file_names[options.startfile:options.endfile]
        inpz=0
        eventperfile=1000
        currentfile=0
        nworkers = 4

        with concurrent.futures.ProcessPoolExecutor(max_workers=nworkers) as executor:
            futures = set()

            futures.update(executor.submit(conversion, filename) for filename in file_names)
        
            #if(len(futures)==0): 
                #continue
        
            '''for k,v in tree_dict:
                futures.update(executor.submit(conversion(v), k, output_directory+k))
                if(len(futures)==0): 
                    continue'''   
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
                print("Ok quitter")
                for job in futures: job.cancel()
            except:
                for job in futures: job.cancel()
                raise

###### old code, non parallelized ######

        # for ifile in file_names:
        #     if currentfile<options.startfile:
        #         currentfile+=1
        #         continue
        #     events = NanoEventsFactory.from_root(ifile, schemaclass=NanoAODSchema).events()
        #     nevents_total = len(events)
        #     print(ifile, ' Number of events:', nevents_total)
            
        #     for i in range(int(nevents_total / eventperfile)+1):
        #         if i< int(nevents_total / eventperfile):
        #             print('from ',i*eventperfile, ' to ', (i+1)*eventperfile)
        #             events_slice = events[i*eventperfile:(i+1)*eventperfile]
        #         elif i == int(nevents_total / eventperfile) and i*eventperfile<=nevents_total:
        #             print('from ',i*eventperfile, ' to ', nevents_total)
        #             events_slice = events[i*eventperfile:nevents_total]
        #         else:
        #             print(' weird ... ')

        #         nparticles_per_event = max(ak.num(events_slice.PFCands.pt, axis=1))
        #         print("max NPF in this range: ", nparticles_per_event)
        #         tic=time.time()
        #         future_savez(i, nevents_total) 
        #         toc=time.time()
        #         print('time:',toc-tic)
        #     currentfile+=1
        #     if currentfile>=options.endfile:
        #         print('=================> finished ')
        #         exit()