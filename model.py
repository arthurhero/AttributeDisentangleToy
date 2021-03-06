import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class ADNet(nn.Module):
    def __init__(self,in_channels,num_atrs):
        '''
        '''
        super(ADNet, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(in_channels,64,kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_ftrs, 1024)
        self.atr_vec = nn.Parameter(torch.randn(1024,num_atrs))

    def forward(self, x):
        '''
        x - b x c x h x w, imgs
        '''
        b = x.shape[0]
        feat_vec = self.resnet50(x) # b x 1024
        with torch.no_grad():
            self.atr_vec = nn.Parameter(self.atr_vec / (self.atr_vec.pow(2).sum(0,keepdim=True).pow(0.5)+10e-8)) # normalize to unit vec
        sim_vec = feat_vec.unsqueeze(1).bmm(self.atr_vec.unsqueeze(0).expand(b,-1,-1)) # b x 1 x 102
        return torch.sigmoid(sim_vec.squeeze(1)) # b x 102
