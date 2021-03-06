# figuring out what the best H/E energy ratio cut should be for Ele and ChPi

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import h5py as h5
import numpy as np
from scipy.stats import binned_statistic

ele_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/EleEscan_*_MERGED/EleEscan_*.h5"
chpi_path = "/u/sciteam/zhang10/Projects/DNNCalorimeter/Data/NewSamples/Fixed/ChPiEscan_*_MERGED/ChPiEscan_*.h5"

###########
# COMBINE #
###########

ele_files = glob.glob(ele_path)
chpi_files = glob.glob(chpi_path)

ele_ratio = []
ele_E = []
chpi_ratio = []
chpi_E = []

for file_name in ele_files:
    file = h5.File(file_name, 'r')
    ele_ratio += list(file['HCAL_ECAL_ERatio'][:])
    ele_E += list(file['energy'][:])
    file.close()

for file_name in chpi_files:
    file = h5.File(file_name, 'r')
    chpi_ratio += list(file['HCAL_ECAL_ERatio'][:])
    chpi_E += list(file['energy'][:])
    file.close()

plt.hist(ele_ratio, bins=np.arange(0,1,0.01), density=True, histtype='step', label='Ele')
plt.hist(chpi_ratio, bins=np.arange(0,1,0.01), density=True, histtype='step', label='ChPi')
plt.xlabel('HCAL_ECAL_ERatio')
plt.ylabel('Normalized Fraction')
plt.legend()
plt.savefig('ratios.png')

plt.clf()

bin_ele_E_means = binned_statistic(ele_ratio, ele_E, bins=50, range=(0, 5)).statistic
bin_chpi_E_means = binned_statistic(chpi_ratio, chpi_E, bins=50, range=(0, 5)).statistic

plt.plot(np.arange(0,5,0.1), bin_ele_E_means, label='Ele')
plt.plot(np.arange(0,5,0.1), bin_chpi_E_means, label='ChPi')
plt.title('Mean Energy in H/E Energy Ratio Bins')
plt.xlabel('HCAL_ECAL_ERatio')
plt.ylabel('Energy')
plt.legend()
plt.savefig('ratio_vs_energy.png')
