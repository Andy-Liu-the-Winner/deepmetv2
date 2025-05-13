from coffea.nanoevents import NanoEventsFactory
from coffea.nanoevents.schemas import PFNanoAODSchema,BaseSchema
import numpy as np
from optparse import OptionParser
import awkward as ak
import time
import json
from collections import OrderedDict,defaultdict
recdd = lambda : defaultdict(recdd) ## define recursive defaultdict
JSON_LOC = 'filelist.json'


def multidict_tojson(filepath, indict):
    ## expand into multidimensional dictionary
    with open(filepath, "w") as fo:
        json.dump( indict, fo)
        print("save to %s" %filepath)


def delta_phi(obj1, obj2):
    return (obj1.phi - obj2.phi + np.pi) % (2 * np.pi) - np.pi

def delta_r(obj1, obj2):
    return np.sqrt((obj1.eta - obj2.eta) ** 2 + delta_phi(obj1, obj2) ** 2)

def run_deltar_matching(store,
                        target,
                        drname='deltaR',
                        radius=0.4,
                        unique=False,
                        sort=False):
  """
  Running a delta R matching of some object collection "store" of dimension NxS
  with some target collection "target" of dimension NxT, The return object will
  have dimension NxSxT' where objects in the T' contain all "target" objects
  within the delta R radius. The delta R between the store and target object will
  be stored in the field `deltaR`. If the unique flag is turned on, then objects
  in the target collection will only be associated to the closest object. If the
  sort flag is turned on, then the target collection will be sorted according to
  the computed `deltaR`.
  """
  _, target = ak.unzip(ak.cartesian([store.eta, target], nested=True))
  target[drname] = delta_r(store, target)
  if unique:  # Additional filtering
    t_index = ak.argmin(target[drname], axis=-2)
    s_index = ak.local_index(store.eta, axis=-1)
    _, t_index = ak.unzip(ak.cartesian([s_index, t_index], nested=True))
    target = target[s_index == t_index]

  # Cutting on the computed delta R
  target = target[target[drname] < radius]

  # Sorting according to the computed delta R
  if sort:
    idx = ak.argsort(target[drname], axis=-1)
    target = target[idx]
  return target


def future_savez(dataset,currentfile):
    print("before selection : " + str(len(events_slice)) + " events")

    #select muons and electrons

    temp = events_slice[:]
    
    tightMuonMask = ((events_slice.Muon.tightId == 1) & ( events_slice.Muon.pfRelIso03_all < 0.15) & (events_slice.Muon.pt > 20.))
    tightElectronMask = ( (events_slice.Electron.pt > 20.0))
   
    events_slice['istightMuon'] = tightMuonMask
    events_slice['istightElectron'] = tightElectronMask
   
    events_slice['Muon'] = events_slice.Muon[events_slice.istightMuon]
    events_slice['Electron'] = events_slice.Electron[events_slice.istightElectron]
    n_tight_leptons = ak.count(events_slice.Muon.pt,axis=-1) + ak.count(events_slice.Electron.pt,axis=-1)
    select_events_mask = n_tight_leptons >= options.n_leptons
    selected_events = events_slice[select_events_mask]

    tightMuonMask = ((selected_events.Muon.tightId == 1) & (selected_events.Muon.pfRelIso03_all < 0.15) & (selected_events.Muon.pt > 20.))
    selected_events['istightMuon'] = tightMuonMask

    tightElectronMask = ((selected_events.Electron.pt > 20.0))
    selected_events['istightElectron'] = tightElectronMask

    muons = selected_events.Muon[selected_events.istightMuon]
    electrons = selected_events.Electron[selected_events.istightElectron]

    leptons = ak.concatenate([muons,electrons],axis=1)
    leptons = leptons[ak.argsort(leptons.pt,axis=1,ascending=False)]

    #only want the first n_leptons_subtract leptons
    leptons_px = leptons.pt * np.cos(leptons.phi)
    leptons_py = leptons.pt * np.sin(leptons.phi)
    leptons_px = ak.sum(leptons_px,axis=1)
    leptons_py = ak.sum(leptons_py,axis=1)

    met_list = np.column_stack([
        ak.to_numpy((selected_events.GenMET.pt * np.cos(selected_events.GenMET.phi) + leptons_px).compute()),
        ak.to_numpy((selected_events.GenMET.pt * np.sin(selected_events.GenMET.phi) + leptons_py).compute()),
        ak.to_numpy((selected_events.MET.pt * np.cos(selected_events.MET.phi) + leptons_px).compute()),
        ak.to_numpy((selected_events.MET.pt * np.sin(selected_events.MET.phi) + leptons_py).compute()),
        ak.to_numpy((selected_events.PuppiMET.pt * np.cos(selected_events.PuppiMET.phi) + leptons_px).compute()),
        ak.to_numpy((selected_events.PuppiMET.pt * np.sin(selected_events.PuppiMET.phi) + leptons_py).compute()),
        ak.to_numpy((selected_events.DeepMETResponseTune.pt * np.cos(selected_events.DeepMETResponseTune.phi) + leptons_px).compute()),
        ak.to_numpy((selected_events.DeepMETResponseTune.pt * np.sin(selected_events.DeepMETResponseTune.phi) + leptons_py).compute()),
        ak.to_numpy((selected_events.DeepMETResolutionTune.pt * np.cos(selected_events.DeepMETResolutionTune.phi) + leptons_px).compute()),
        ak.to_numpy((selected_events.DeepMETResolutionTune.pt * np.sin(selected_events.DeepMETResolutionTune.phi) + leptons_py).compute()),
        ak.to_numpy(selected_events.LHE.HT.compute())
    ])
    met_list=np.array(met_list)
    # print("met_list:", met_list, "\n")
    print("met_list shape:", met_list.shape, "\n")

    
    
    overlap_removal = run_deltar_matching(selected_events.PFCands,
                        leptons,
                        drname='deltaR',
                        radius=0.001,
                        unique=True,
                        sort=False)
    
    mask = ak.count(overlap_removal.deltaR,axis=-1)==0
    # remove the cloest PF particle 
    mask = ak.count(overlap_removal.deltaR,axis=-1)==0
    #print(len(selected_events.PFCands.pt[0]))
    selected_events['PFCands']=selected_events.PFCands[mask]
    #print(len(selected_events.PFCands.pt[0]))

    nparticles_per_event = int(max(ak.num(selected_events.PFCands.pt, axis=1).compute()))
    
    # print("nparticles_per_event:", nparticles_per_event, "\n")
    # print(len(selected_events.compute()))
    #save the rest of PFcandidates 
    particle_list = np.full((11,len(selected_events.compute()),nparticles_per_event),-999, dtype='float32')
    particle_list[0]= ak.fill_none(ak.pad_none(selected_events.PFCands.pt, nparticles_per_event,clip=True).compute(),-999)
    particle_list[1]= ak.fill_none(ak.pad_none(selected_events.PFCands.eta, nparticles_per_event,clip=True).compute(),-999)
    particle_list[2]= ak.fill_none(ak.pad_none(selected_events.PFCands.phi, nparticles_per_event,clip=True).compute(),-999)
    particle_list[3]= ak.fill_none(ak.pad_none(selected_events.PFCands.d0, nparticles_per_event,clip=True).compute(),-999)
    particle_list[4]= ak.fill_none(ak.pad_none(selected_events.PFCands.dz, nparticles_per_event,clip=True).compute(),-999)
    particle_list[5]= ak.fill_none(ak.pad_none(selected_events.PFCands.mass, nparticles_per_event,clip=True).compute(),-999)
    particle_list[6]= ak.fill_none(ak.pad_none(selected_events.PFCands.puppiWeight, nparticles_per_event,clip=True).compute(),-999)
    particle_list[7]= ak.fill_none(ak.pad_none(selected_events.PFCands.pdgId, nparticles_per_event,clip=True).compute(),-999)
    particle_list[8]= ak.fill_none(ak.pad_none(selected_events.PFCands.charge, nparticles_per_event,clip=True).compute(),-999)
    particle_list[9]= ak.fill_none(ak.pad_none(selected_events.PFCands.fromPV, nparticles_per_event,clip=True).compute(),-999)
    # particle_list[10]= ak.fill_none(ak.pad_none(selected_events.PFCands.pvRef, nparticles_per_event,clip=True),-999)
    particle_list[10]= ak.fill_none(ak.pad_none(selected_events.PFCands.pvAssocQuality, nparticles_per_event,clip=True).compute(),-999)
    print("particle_list shape:", particle_list.shape, "\n")
   
    
    print("saving")
    
    
    npz_file='/hildafs/projects/phy230010p/share/NanoAOD/npz_files/'+dataset+'/raw/'+dataset+'_file'+str(currentfile)+'_slice_'+str(i)+'_nevent_'+str(len(selected_events.compute()))
    
    #,y=met_list

    np.savez(npz_file,x=particle_list,y=met_list)





if __name__ == '__main__':
        
        parser = OptionParser()
        parser.add_option('-d', '--dataset', help='dataset', dest='dataset')
        parser.add_option('-s', '--startfile',type=int, default=0, help='startfile')
        parser.add_option('-e', '--endfile',type=int, default=1, help='endfile')
        parser.add_option('--n_leptons', dest='n_leptons',
                          help='How many leptons are required in the events', default=2)
        parser.add_option('--n_leptons_subtract', dest='n_leptons_subtract',
                          help='How many leptons to be subtracted from the Candidates list. Can not be larger than the n_leptons', default=2)
        parser.add_option('--nevents', help='number of events to convert', dest='eventsmax', default=300000)
        (options, args) = parser.parse_args()

        
        assert options.n_leptons >= options.n_leptons_subtract, "n_leptons_subtract can not be larger than n_leptons"

        dataset=options.dataset

        datasetsname = {
            "dy",
            "dyold",
            "tt"
        }

        if dataset not in datasetsname:
            print('choose one of them: ', datasetsname)
            exit()

        #Read file from json
        with open(JSON_LOC, "r") as fo:
            file_names = json.load(fo)
        file_names = file_names[dataset]
        print('found ', len(file_names)," files")
        if options.startfile>=options.endfile and options.endfile!=-1:
            print("make sure options.startfile<options.endfile")
            exit()
        inpz=0
        eventperfile=5000
        currentfile=0
        eventsmax = int(options.eventsmax)
        eventsconverted=0
        for file in file_names:
            if currentfile<options.startfile:
                currentfile+=1
                continue
            events = NanoEventsFactory.from_root({file: 'Events'}, schemaclass=PFNanoAODSchema).events()
            nevents_total = len(events.compute())
            print(file, ' Number of events:', nevents_total)

            for i in range(int(nevents_total / eventperfile)+1):
                if i< int(nevents_total / eventperfile):
                    print('from ',i*eventperfile, ' to ', (i+1)*eventperfile)
                    events_slice = events[i*eventperfile:(i+1)*eventperfile]
                elif i == int(nevents_total / eventperfile) and i*eventperfile<=nevents_total:
                    print('from ',i*eventperfile, ' to ', nevents_total)
                    events_slice = events[i*eventperfile:nevents_total]
                else:
                    print(' weird ... ')
                    exit()
                tic=time.time()
                future_savez(dataset,currentfile)
                toc=time.time()
                print('time:',toc-tic)
                if eventsconverted>=eventsmax:
                    print('=================> finished ')
                    exit()
            currentfile+=1
            if currentfile>=options.endfile:
                print('=================> finished ')
                exit()
