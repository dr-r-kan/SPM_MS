# load the microstatelab template set file and export a matlab .mat file for easier handling

file = r"/home/rohan/PycharmProjects/SPM_MS/Metamaps MS Template/MetaMaps_2023_06.set"

import mne

e = mne.io.read_raw_eeglab(file, preload=True)
print(e.info)

print(e.get_data().shape)

Channels = e.info['ch_names']

# convert the data to a matlab .mat file - with the channels as rows and timepoints as columns, have the channel names and digitization points too (x,y,z)
import scipy.io as sio
import numpy as np
data = e.get_data()[:, 15:22] # get the data for the 7 microstates only (not the abridged groups)
sio.savemat("MetaMaps_2023_06.mat", {'data': data, 'Channels': Channels})
# save the channel locations too
montage = e.get_montage()
if montage is not None:
    dig = montage.get_positions()['ch_pos']
    dig_array = np.array([list(dig[ch]) for ch in Channels if ch in dig])
    sio.savemat("MetaMaps_2023_06_dig.mat", {'dig': dig_array})


# for each of the 22 samples in the dataset, plot the topomaps all on the same scale and axis
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10, 5))
for i in range(7):
    ax = fig.add_subplot(1, 7, i+1)
    mne.viz.plot_topomap(data[:, i], e.info, axes=ax, show=False)
    ax.set_title(f'MS {i+1}')
plt.tight_layout()
plt.show()