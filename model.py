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
        self.resnet50.fc = nn.Linear(1024, 1024)
        self.atr_vec = nn.Parameter(torch.randn(1024,num_atrs))
        self.q_encoder = nn.Linear(1024,1024)
        self.k_encoder = nn.Linear(1024,1024)
        self.attn = nn.MultiheadAttention(1024, num_heads=1)

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
                last_feat = F.interpolate(last_feat,(32,32),mode='bilinear').reshape(b,2048,-1).permute(1,0,2) # 2048 x b x 1024
                query = self.q_encoder(self.atr_vec.transpose(0,1).unsqueeze(1).expand(-1,b,-1)) # num_atrs x b x 1024
                kv = self.k_encoder(last_feat) # 2048 x b x 1024
                attended,_ = self.attn(query,kv,kv) # num_atrs x b x 1024
                x = attended.mean(0) # b x 1024 
        feat_vec = x # b x 1024
        sim_vec = feat_vec.unsqueeze(1).bmm(self.atr_vec.unsqueeze(0).expand(b,-1,-1)) # b x 1 x 102
        return torch.sigmoid(sim_vec.squeeze(1)) # b x 102
