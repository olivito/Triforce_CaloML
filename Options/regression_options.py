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
options['nEpochs'] = 5 # break after this number of epochs
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
from Regression import NIPS_Regressor,DNNCellsSumSkipRegressor,CNNCellsSumSkipRegressor
from GAN import NIPS_GAN
from Analysis import Default_Analyzer

_learningRate = 0.001
_decayRate = 0.01
_dropoutProb = 0.2
_hiddenLayerNeurons = 512
_nHiddenLayers = 2

#classifier = GoogLeNet.Classifier(_learningRate, _decayRate)
classifier = None
#regressor = NIPS_Regressor.Regressor(_learningRate, _decayRate)
regressor = DNNCellsSumSkipRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = CNNCellsSumSkipRegressor.Regressor(_dropoutProb, _learningRate, _decayRate)
GAN = None
analyzer = Default_Analyzer.Analyzer()
