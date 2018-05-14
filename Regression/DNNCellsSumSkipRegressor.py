import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##############
# Regression #
##############

class Regressor_Net(nn.Module):
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb):
        super().__init__()
        self.ECALsize = 51
        self.HCALsize = 11
        self.input = nn.Linear(self.ECALsize * self.ECALsize * 25 + self.HCALsize * self.HCALsize * 60 + 2, hiddenLayerNeurons)
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.nHiddenLayers = nHiddenLayers
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons+2, 1)
        # initialize last two weights in output layer to 1: assume close to identity for energy sums
        output_params = self.output.weight.data
        output_params[0][-1] = 1.0
        output_params[0][-2] = 1.0
    def forward(self, x1, x2):
        x1 = x1.view(-1, self.ECALsize * self.ECALsize * 25)
        x2 = x2.view(-1, self.HCALsize * self.HCALsize * 60)
        x1_sum = torch.sum(x1, dim = 1).view(-1, 1)
        x2_sum = torch.sum(x2, dim = 1).view(-1, 1)
        x = torch.cat([x1, x2, x1_sum, x2_sum], 1)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        # add ECAL and HCAL sums as inputs to last layer
        x = torch.cat([x,x1_sum,x2_sum],1)
        x = self.output(x)
        return x
class Regressor():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Regressor_Net(hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.MSELoss()
