import os, sys
options = {}

##################
# Choose samples #
##################

basePath = "/data/shared/LCDLargeWindow/fixedangle/"
options['samplePath'] = [basePath + "EleEscan/EleEscan_*.h5"]
options['classPdgID'] = [11] # absolute IDs corresponding to paths above
options['eventsPerFile'] = 10000
options['nWorkers'] = 2

###############
# Job options #
###############

options['trainRatio'] = 0.66
options['nEpochs'] = 50 # break after this number of epochs
options['relativeDeltaLossThreshold'] = 0.0 # break if change in loss falls below this threshold over an entire epoch, or...
options['relativeDeltaLossNumber'] = 5 # ...for this number of test losses in a row
options['batchSize'] = 200 # 1000
options['saveModelEveryNEpochs'] = 0 # 0 to only save at end
options['nTrainMax'] = -1
options['nTestMax'] = -1
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

################
# Choose tools #
################

from Classification import GoogLeNet
from Regression import DNNFeatureSkipRegressor
from GAN import NIPS_GAN
from Analysis import Default_Analyzer_features

_learningRate = 0.001
_decayRate = 0.01
_dropoutProb = 0.2
_hiddenLayerNeurons = 12
_nHiddenLayers = 2
_nInputs = 3

classifier = None
regressor = DNNFeatureSkipRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers,  _learningRate, _decayRate, _nInputs)
GAN = None
analyzer = Default_Analyzer_features.Analyzer()
