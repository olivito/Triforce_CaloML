import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

##################
# Classification #
##################

class Classifier_Net(nn.Module):
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb):
        super().__init__()
        self.input = nn.Linear(51 * 51 * 25 + 1, hiddenLayerNeurons)
        self.input = nn.DataParallel(self.input) # multi-GPU
        self.hidden = nn.Linear(hiddenLayerNeurons, hiddenLayerNeurons)
        self.hidden = nn.DataParallel(self.hidden) # multi-GPU
        self.nHiddenLayers = nHiddenLayers
        self.dropout = nn.Dropout(p = dropoutProb)
        self.output = nn.Linear(hiddenLayerNeurons, 2)
        self.output = nn.DataParallel(self.output) # multi-GPU
    def forward(self, x, _, eta=None):
        x = x.view(-1, 51 * 51 * 25)
        x = x.append(eta)
        x = self.input(x)
        for i in range(self.nHiddenLayers-1):
            x = F.relu(self.hidden(x))
            x = self.dropout(x)
        x = F.softmax(self.output(x), dim=1)
        return x

class Classifier():
    def __init__(self, hiddenLayerNeurons, nHiddenLayers, dropoutProb, learningRate, decayRate):
        self.net = Classifier_Net(hiddenLayerNeurons, nHiddenLayers, dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = nn.CrossEntropyLoss()
