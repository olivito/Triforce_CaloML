import os, sys
options = {}

##################
# Choose samples #
##################

#basePath = "/data/LCD/V3/Original/EleChPi/"
#options['samplePath'] = [basePath + "ChPi/ChPiEscan_*.h5", basePath + "Ele/EleEscan_*.h5"]
#options['classPdgID'] = [211, 11] # absolute IDs corresponding to paths above
#basePath = "/bigdata/shared/LCDLargeWindow/LCDLargeWindow/fixedangle/"
basePath = "/data/shared/LCDLargeWindow/fixedangle/"
basePath = "/data/shared/LCDLargeWindow/varangle/"
options['samplePath'] = [basePath + "EleEscan/EleEscan_*.h5"]
options['classPdgID'] = [11] # absolute IDs corresponding to paths above
#options['samplePath'] = [basePath + "GammaEscan/GammaEscan_*.h5"]
#options['samplePath'] = [basePath + "GammaEscan/skim_conv_2kfiles/GammaEscan_*.h5"]
#options['classPdgID'] = [22] # absolute IDs corresponding to paths above
#options['samplePath'] = [basePath + "ChPiEscan/skim_RecoOverTrueEgt0p3/ChPiEscan_*.h5"]
#options['classPdgID'] = [211] # absolute IDs corresponding to paths above
#options['samplePath'] = [basePath + "Pi0Escan/Pi0Escan_*.h5"]
#options['classPdgID'] = [111] # absolute IDs corresponding to paths above
#basePath = "/bigdata/shared/LCD/V1/"
#options['samplePath'] = [basePath + "EleEscan/train/EleEscan_*.h5"]
#options['classPdgID'] = [11] # absolute IDs corresponding to paths above
options['eventsPerFile'] = 10000
#options['eventsPerFile'] = 2000
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
#options['nTrainMax'] = 10
#options['nTestMax'] = 5
options['outPath'] = os.getcwd()+"/Output/"+sys.argv[1]+"/"

################
# Choose tools #
################

from Classification import GoogLeNet
from Regression import NIPS_Regressor,DNNRegressor,DNNSmallRegressor,DNNSumRegressor,DNNSmallSumRegressor,DNNSmallSumSkipRegressor,DNNSmallLayersSumSkipRegressor,DNNCellsSumSkipRegressor,CNNCellsRegressor,CNNCellsSumSkipRegressor,ResNetSmallECALRegressor,ResNetSmallECALSumSkipRegressor,GoogLeNetSmallECALRegressor
from GAN import NIPS_GAN
from Analysis import Default_Analyzer

_learningRate = 0.001
_decayRate = 0.01
_dropoutProb = 0.2
_hiddenLayerNeurons = 512
#_hiddenLayerNeurons = 4
_nHiddenLayers = 2

#classifier = GoogLeNet.Classifier(_learningRate, _decayRate)
classifier = None
#regressor = NIPS_Regressor.Regressor(_learningRate, _decayRate)
#regressor = DNNRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = DNNSmallRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = DNNSumRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _learningRate, _decayRate)
#regressor = DNNSmallSumRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = DNNSmallSumSkipRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = DNNSmallLayersSumSkipRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
#regressor = CNNCellsRegressor.Regressor(_dropoutProb, _learningRate, _decayRate)

### best architectures
#regressor = DNNCellsSumSkipRegressor.Regressor(_hiddenLayerNeurons, _nHiddenLayers, _dropoutProb, _learningRate, _decayRate)
regressor = CNNCellsSumSkipRegressor.Regressor(_dropoutProb, _learningRate, _decayRate)

### testing
#regressor = ResNetSmallECALRegressor.Regressor(_learningRate, _decayRate)
#regressor = ResNetSmallECALSumSkipRegressor.Regressor(_learningRate, _decayRate)
#regressor = GoogLeNetSmallECALRegressor.Regressor(_learningRate, _decayRate)

GAN = None
analyzer = Default_Analyzer.Analyzer()
