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
        self.cond_mat = nn.Parameter(torch.randn(1024,1024))
        self.running_corr = nn.Parameter(torch.eye(num_atrs).clamp(min=0.5), requires_grad=False) # prior

    def forward(self, x, corr_labels):
        '''
        x - b x c x h x w, imgs
        corr_labels - b x 102 x 102
        '''
        b = x.shape[0]
        feat_vec = self.resnet50(x) # b x 1024
        with torch.no_grad():
            self.atr_vec = nn.Parameter(self.atr_vec / (self.atr_vec.pow(2).sum(0,keepdim=True).pow(0.5)+10e-8)) # normalize to unit vec
        corr_mat = self.atr_vec.transpose(0,1).matmul(self.cond_mat).matmul(self.atr_vec) # num_atrs x num_atrs
        sim_vec = feat_vec.unsqueeze(1).matmul(self.cond_mat).matmul(self.atr_vec) # b x 1 x 102

        self.running_corr = nn.Parameter(self.running_corr + corr_labels.sum(0), requires_grad=False) # 102 x 102
        conds = self.running_corr / self.running_corr.diagonal().unsqueeze(1)
        return torch.sigmoid(sim_vec.squeeze(1)), torch.sigmoid(corr_mat), conds # b x 102, 102 x 102, 102 x 102
