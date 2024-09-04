import pickle
import matplotlib.pyplot as plt
import numpy as np
import argparse
import mplhep as hep
import utils
plt.style.use(hep.style.CMS)

# with open('weight.plt', 'rb') as file:
#     header = file.read(5)
#     print(header)

# with open('weight.plt', 'rb') as file:
#     weights = pickle.load(file)

# Replace 'yourfile.coffea' with the actual file name
weights = utils.load('weight.plt')

pt_x_values = 0.5*(weights['bin_edges']['Pt'][1:] + weights['bin_edges']['Pt'][:-1])

charged_bins = weights['bin_edges']['graph_weight']

pt_y_values = weights['weight_pt_hist']

hadron0_values = weights['weight_CH_hist']['puppi0']['Charged Hadron']

hadron1_values = weights['weight_CH_hist']['puppi1']['Charged Hadron']

muon0_values = weights['weight_CH_hist']['puppi0']['Muon']

muon1_values = weights['weight_CH_hist']['puppi1']['Muon']

electron0_values = weights['weight_CH_hist']['puppi0']['Electron']

electron1_values = weights['weight_CH_hist']['puppi1']['Electron']

particle_list = ['HF Candidate', 'Electron', 'Muon', 'Gamma', 'Neutral Hadron', 'Charged Hadron']

color_dict = {
    'HF Candidate': 'black',
    'Electron': 'red',
    'Muon': 'limegreen',
    'Gamma': 'blue',
    'Neutral Hadron': 'magenta',
    'Charged Hadron': 'cyan'
}

plt.figure(1)
# Plot each category
for particle in particle_list:
    plt.plot(pt_x_values, pt_y_values[particle], marker='o', color=color_dict[particle], label=particle)

# Set x-axis to logarithmic
plt.xscale('log')

# Set domain and range
plt.xlim(0.07, 25)
plt.ylim(0, 1.4)

# Add labels and legend
plt.xlabel(r'PF $P_{T}$ [GeV]')
plt.ylabel('GraphMet Weight')
plt.legend()

# Show the plot
plt.savefig('weights_cropped.png')

plt.figure(2)
plt.stairs(hadron0_values, charged_bins, color='blue', label='puppi==0')
plt.stairs(hadron1_values, charged_bins, color='red', label='puppi==1')
plt.axvline(x=1, linestyle='dashed', color='black')
plt.yscale('log')
plt.xlim(-0.05, 5)
plt.xlabel('GraphMet Weight')
plt.ylabel('Charged Hadron')
plt.legend()
plt.savefig('charged_weights_hadron.png')

plt.figure(3)
plt.stairs(muon0_values, charged_bins, color='blue', label='puppi==0')
plt.stairs(muon1_values, charged_bins, color='red', label='puppi==1')
plt.axvline(x=1, linestyle='dashed', color='black')
plt.yscale('log')
plt.xlim(-0.05, 3)
plt.xlabel('GraphMet Weight')
plt.ylabel('Muon')
plt.legend()
plt.savefig('charged_weights_muon.png')

plt.figure(4)
plt.stairs(electron0_values, charged_bins, color='blue', label='puppi==0')
plt.stairs(electron1_values, charged_bins, color='red', label='puppi==1')
plt.axvline(x=1, linestyle='dashed', color='black')
plt.yscale('log')
plt.xlim(-0.05, 2)
plt.xlabel('GraphMet Weight')
plt.ylabel('Electron')
plt.legend()
plt.savefig('charged_weights_electron.png')
