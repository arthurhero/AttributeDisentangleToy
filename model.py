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
        #num_ftrs = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_atrs*49, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.atr_vec = nn.Parameter(torch.randn(1024,num_atrs))
        self.q_encoder = nn.Linear(1024,1024)
        self.k_encoder = nn.Linear(49,1024)

    def forward(self, x):
        '''
        x - b x c x h x w, imgs
        '''
        b = x.shape[0]
        with torch.no_grad():
            self.atr_vec = nn.Parameter(self.atr_vec / (self.atr_vec.pow(2).sum(0,keepdim=True).pow(0.5)+10e-8)) # normalize to unit vec
        for k, v in self.resnet50._modules.items():
            if 'avgpool' in k:
                continue
            x = v(x)
            if 'layer4' in k:
                last_feat = x.clone() # b x 2048 x h' x w'
                last_feat = F.interpolate(last_feat,(7,7),mode='bilinear').reshape(b,2048,-1) # b x 2048 x 49
                key = self.k_encoder(last_feat) # b x 2048 x 1024 
                query = (self.q_encoder(self.atr_vec.transpose(0,1))).transpose(0,1).unsqueeze(0).expand(b,-1,-1) # b x 1024 x num_atrs
                weights = F.softmax(torch.bmm(key,query), dim=1) # b x 2048 x num_atrs 
                attended = last_feat.unsqueeze(2) * weights.unsqueeze(3) # b x 2048 x num_atrs x 49
                attended = attended.sum(1) # b x num_atrs x 49
                x = attended.reshape(b,-1) # b x (num_atrs x 49)
        feat_vec = self.fc2(x) # b x 1024
        sim_vec = feat_vec.unsqueeze(1).bmm(self.atr_vec.unsqueeze(0).expand(b,-1,-1)) # b x 1 x 102
        return torch.sigmoid(sim_vec.squeeze(1)) # b x 102
