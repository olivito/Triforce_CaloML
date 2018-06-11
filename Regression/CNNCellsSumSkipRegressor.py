import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import Regression.loss_functions as loss_functions

##############
# Regression #
##############

class Regressor_Net(nn.Module):
    def __init__(self, dropoutProb):
        super().__init__()
        self.ECALsize = 51
        self.HCALsize = 11
        self.ECALnconv = 3
        self.ECALconvXY = 4
        self.ECALconvZ = 4
        self.HCALnconv = 3
        self.HCALconvXY = 2
        self.HCALconvZ = 6
        self.conv1 = nn.Conv3d(1, self.ECALnconv, (self.ECALconvXY, self.ECALconvXY, self.ECALconvZ))
        self.conv2 = nn.Conv3d(1, self.HCALnconv, (self.HCALconvXY, self.HCALconvXY, self.HCALconvZ))
        self.dropout = nn.Dropout(p = dropoutProb)
#        self.linear1 = nn.Linear(5073+2, 1000) # NIPS settings # 3 ECALnconv, 10 HCALnconv, NIPS window of 25 ECAL, 5 HCAL
#        self.linear1 = nn.Linear(3993+324+2, 1000) # 3 ECALnconv, 3 HCALnconv, NIPS window of 25 ECAL, 5 HCAL
#        self.linear1 = nn.Linear(13310+324+2, 1000) # 10 ECALnconv, 3 HCALnconv, NIPS window of 25 ECAL, 5 HCAL
        self.linear1 = nn.Linear(19008+2025+2, 1000) # 3 ECALnconv, 3 HCALnconv, bigger window of 51 ECAL, 11 HCAL <-- new default
#        self.linear1 = nn.Linear(63360+2025+2, 1000) # 10 ECALnconv, 3 HCALnconv, bigger window of 51 ECAL, 11 HCAL
        self.output = nn.Linear(1000+2, 1)
        # initialize last two weights in linear and output layer to 1: assume close to identity for energy sums
        linear1_params = self.linear1.weight.data
        linear1_params[0][-1] = 1.0
        linear1_params[0][-2] = 1.0
        output_params = self.output.weight.data
        output_params[0][-1] = 1.0
        output_params[0][-2] = 1.0
    def forward(self, ECALs, HCALs):
        # ECAL input
        r = ECALs.view(-1, 1, self.ECALsize, self.ECALsize, 25)
        ECALs_sum = torch.sum(ECALs.view(ECALs.size(0),-1), dim = 1).view(-1, 1)
        model1 = self.conv1(r)
        model1 = F.relu(model1)
        model1 = nn.MaxPool3d(2)(model1)
        #print(model1.size())
        model1 = model1.view(model1.size(0), -1)
        #print(model1.size())
        # HCAL input
        r = HCALs.view(-1, 1, self.HCALsize, self.HCALsize, 60)
        HCALs_sum = torch.sum(HCALs.view(HCALs.size(0),-1), dim = 1).view(-1, 1)
        model2 = self.conv2(r)
        model2 = F.relu(model2)
        model2 = nn.MaxPool3d(2)(model2)
        #print(model2.size())
        model2 = model2.view(model2.size(0), -1)
        #print(model2.size())
        # join the two input models
        bmodel = torch.cat((model1, model2, ECALs_sum, HCALs_sum), 1)  # branched model
        # fully connected ending
        bmodel = self.linear1(bmodel)
        bmodel = F.relu(bmodel)
        bmodel = self.dropout(bmodel)
        bmodel = torch.cat((bmodel, ECALs_sum, HCALs_sum), 1)
        bmodel = self.output(bmodel)
        return bmodel

class Regressor():
    def __init__(self, dropoutProb, learningRate, decayRate):
        self.net = Regressor_Net(dropoutProb)
        self.net.cuda()
        self.optimizer = optim.Adam(self.net.parameters(), lr=learningRate, weight_decay=decayRate)
        self.lossFunction = loss_functions.weighted_mse_loss
#        self.lossFunction = nn.MSELoss()
