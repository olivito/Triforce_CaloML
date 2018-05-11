# HDF5Dataset is of class torch.utils.data.Dataset, and is initialized with a set of data files and the number of events per file.
# __len__ returns the number of items in the dataset, which is simply the number of files times the number of events per file.
# __getitem__ takes an index and returns that event. First it sees which file the indexed event would be in, and loads that file if it is not already in memory. It reads the entire ECAL, HCAL, and target information of that file into memory. Then it returns info for the requested event.
# OrderedRandomSampler is used to pass indices to HDF5Dataset, but the indices are created in such a way that the first file is completely read first, and then the second file, then the third etc.

import torch.utils.data as data
from torch import from_numpy
import h5py
import numpy as np

def load_hdf5(file):

    """Loads H5 file. Used by HDF5Dataset."""

    with h5py.File(file, 'r') as f:
        ECAL_E = f['ECAL_E'][:].reshape(-1,1)
        HCAL_E = f['HCAL_E'][:].reshape(-1,1)
        ECALmomentX2 = f['ECALmomentX2'][:].reshape(-1,1)
        ECALmomentZ1 = f['ECALmomentZ1'][:].reshape(-1,1)
        HCALmomentX2 = f['HCALmomentX2'][:].reshape(-1,1)
        HCALmomentY2 = f['HCALmomentY2'][:].reshape(-1,1)
        HCALmomentXY2 = np.sqrt(np.square(HCALmomentX2) + np.square(HCALmomentY2))
        HCALmomentZ1 = f['HCALmomentZ1'][:].reshape(-1,1)
        ## select which features to use here.  Regressor assumes ECAL_E and HCAL_E are last two features
        #features = np.concatenate([ECALmomentX2, ECALmomentZ1, HCALmomentXY2, HCALmomentZ1, ECAL_E, HCAL_E], axis=1)
        features = np.concatenate([ECALmomentZ1, ECAL_E, HCAL_E], axis=1)
        #features = np.concatenate([ECAL_E, HCAL_E], axis=1)
        pdgID = f['pdgID'][:]
        energy = f['energy'][:]

    return features.astype(np.float32), pdgID.astype(int), energy.astype(np.float32)

class HDF5Dataset(data.Dataset):

    """Creates a dataset from a set of H5 files.
        Used to create PyTorch DataLoader.
    Arguments:
        dataname_tuples: list of filename tuples, where each tuple will be mixed into a single file
        num_per_file: number of events in each data file
    """

    def __init__(self, dataname_tuples, num_per_file, classPdgID):
        self.dataname_tuples = sorted(dataname_tuples)
        self.num_per_file = num_per_file
        self.fileInMemory = -1
        self.features = []
        self.y = []
        self.classPdgID = {}
        for i, ID in enumerate(classPdgID):
            self.classPdgID[ID] = i

    def __getitem__(self, index):
        fileN = index//self.num_per_file
        indexInFile = index%self.num_per_file-1
        if(fileN != self.fileInMemory):
            self.features = []
            self.y = []
            for dataname in self.dataname_tuples[fileN]:
                file_features, file_pdgID, energy = load_hdf5(dataname)
                if len(file_pdgID.shape) == 2: # in case this has the wrong dimensions
                    file_pdgID = file_pdgID[:,0]
                if (self.features != []):
                    self.features = np.append(self.features, file_features, axis=0)
                    newy = [self.classPdgID[abs(i)] for i in file_pdgID] # should probably make this one-hot
                    self.y = np.append(self.y, newy) 
                    self.energy = np.append(self.energy, energy)
                else:
                    self.features = file_features
                    self.y = [self.classPdgID[abs(i)] for i in file_pdgID] # should probably make this one-hot
                    self.energy = energy
            self.fileInMemory = fileN
        return self.features[indexInFile], self.y[indexInFile], self.energy[indexInFile]

    def __len__(self):
        return len(self.dataname_tuples)*self.num_per_file

class OrderedRandomSampler(object):

    """Samples subset of elements randomly, without replacement.
    Arguments:
        data_source (Dataset): dataset to sample from
    """

    def __init__(self, data_source):
        self.data_source = data_source
        self.num_per_file = self.data_source.num_per_file
        self.num_of_files = len(self.data_source.dataname_tuples)

    def __iter__(self):
        indices=np.array([],dtype=np.int64)
        for i in range(self.num_of_files):
            indices=np.append(indices, np.random.permutation(self.num_per_file)+i*self.num_per_file)
        return iter(from_numpy(indices))

    def __len__(self):
        return len(self.data_source)
