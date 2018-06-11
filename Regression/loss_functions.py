import torch

def weighted_mse_loss(input,target,weights):
    out = (input.view(-1)-target)**2
    #print(input.size(),target.size(),out.size())
    out = out * weights
    loss = torch.sum(out, dim=0) 
    #print(out.size(),weights.size(),loss.size())
    return loss
