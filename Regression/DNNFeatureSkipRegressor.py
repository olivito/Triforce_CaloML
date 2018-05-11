import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############
# Regression #
##############

class Regressor_Net(nn.Module):
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, nInputs):
        super().__init__()
        self.input = nn.Linear(nInputs, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.nHiddenLayers = nHiddenLayers
        self.output = nn.Linear(hiddenLayerNeurons+nInputs, 1)
        # initialize last two weights in output layer to 1: assume close to identity for energy sums
        # also assumes that ECAL_E, HCAL_E are the last two features.
        output_params = self.output.weight.data
        output_params[0][-1] = 1.0
        output_params[0][-2] = 1.0
    def forward(self, x):
        y = self.input(x)
        for i in range(self.nHiddenLayers-1):
            y = F.relu(self.hidden(y))
        # skip connection for last layer
        y = torch.cat([y,x],1)
        y = self.output(y)
        return y
class Regressor():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, learningRate, decayRate, nInputs):
        self.net = Regressor_Net(hiddenLayerNeurons, nHiddenLayers, nInputs)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.MSELoss()
