from utils import load
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import mplhep as hep
plt.style.use(hep.style.CMS)

parser = argparse.ArgumentParser()
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before \
                    training")  # 'best' or 'train'
parser.add_argument('--ckpts', default='ckpts',
                    help="Name of the ckpts folder")
parser.add_argument('--comparison', default=None, help="Name of ckpts folder to compare with")


args = parser.parse_args()
a=load(args.ckpts + '/' +args.restore_file+ '.resolutions')
colors = {
    'pfMET': 'black',
    'puppiMET': 'red',
    'deepMETResponse': 'blue',
    'deepMETResolution': 'green',
    'MET':  'magenta',
}
label_arr = {
    'MET':     'DeepMETv2' ,
    'pfMET':    'PF MET',
    'puppiMET': 'PUPPI MET',
    'deepMETResponse': 'DeepMETResponse',
    'deepMETResolution': 'DeepMETResolution',
}
resolutions_arr = {
    'MET':      [[],[],[]],
    'pfMET':    [[],[],[]],
    'puppiMET': [[],[],[]],
    'deepMETResponse': [[],[],[]],
    'deepMETResolution': [[],[],[]],
}
for key in resolutions_arr:
         #skip deepMETResolution and deepMETResponse while broken
         if key == 'deepMETResolution' or key == 'deepMETResponse':
              continue
         plt.figure(1)
        #  print(a[key]['u_perp_resolution'][1])
        #  print(a[key]['u_perp_resolution'][0])
         if key == 'deepMETResponse':
            print(a[key]['u_perp_resolution'][1][0:40])
            print(a[key]['u_perp_resolution'][0][0:40])
         if key == 'MET':
            print(a[key]['u_perp_resolution'][1][0:40])
            print(a[key]['u_perp_resolution'][0][0:40])
         xx = a[key]['u_perp_resolution'][1][0:40]
        #  print(xx.shape)
         yy = a[key]['u_perp_resolution'][0][0:40]
        #  print(yy.shape)
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(2)
         xx = a[key]['u_perp_scaled_resolution'][1][0:40]
         yy = a[key]['u_perp_scaled_resolution'][0][0:40]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(3)
         xx = a[key]['u_par_resolution'][1][0:40]
         yy = a[key]['u_par_resolution'][0][0:40]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(4)
         xx = a[key]['u_par_scaled_resolution'][1][0:40]
         yy = a[key]['u_par_scaled_resolution'][0][0:40]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])
         plt.figure(5)
         xx = a[key]['R'][1][0:40]
         yy = a[key]['R'][0][0:40]
         plt.plot(xx, yy,color=colors[key], label=label_arr[key])

if args.comparison != None:
    a = load(args.comparison + '/best.resolutions')
    plt.figure(1)
    xx = a['MET']['u_perp_resolution'][1][0:40]
    #  print(xx.shape)
    yy = a['MET']['u_perp_resolution'][0][0:40]
    #  print(yy.shape)
    plt.plot(xx, yy,color='brown', label='comparison')
    plt.figure(2)
    xx = a['MET']['u_perp_scaled_resolution'][1][0:40]
    yy = a['MET']['u_perp_scaled_resolution'][0][0:40]
    plt.plot(xx, yy,color='brown', label='comparison')
    plt.figure(3)
    xx = a['MET']['u_par_resolution'][1][0:40]
    yy = a['MET']['u_par_resolution'][0][0:40]
    plt.plot(xx, yy,color='brown', label='comparison')
    plt.figure(4)
    xx = a['MET']['u_par_scaled_resolution'][1][0:40]
    yy = a['MET']['u_par_scaled_resolution'][0][0:40]
    plt.plot(xx, yy,color='brown', label='comparison')
    plt.figure(5)
    xx = a['MET']['R'][1][0:40]
    yy = a['MET']['R'][0][0:40]
    plt.plot(xx, yy,color='brown', label='comparison')

if(True):
    model_dir=args.ckpts+'/'+args.restore_file+'_'
    plt.figure(1)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp.png')
    plt.clf()
    plt.close()

    plt.figure(2)
    plt.axis([0, 400, 0, 35])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\perp})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_perp_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(3)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'$\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel.png')
    plt.clf()
    plt.close()

    plt.figure(4)
    plt.axis([0, 400, 0, 60])
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Scaled $\sigma (u_{\parallel})$ [GeV]')
    plt.legend()
    plt.savefig(model_dir+'resol_parallel_scaled.png')
    plt.clf()
    plt.close()

    plt.figure(5)
    plt.axis([0, 400, 0, 1.2])
    plt.axhline(y=1.0, color='black', linestyle='-.')
    plt.xlabel(r'$q_{T}$ [GeV]')
    plt.ylabel(r'Response $-\frac{<u_{\parallel}>}{<q_{T}>}$')
    plt.legend()
    plt.savefig(model_dir+'response_parallel.png')
    plt.clf()
    plt.close()



