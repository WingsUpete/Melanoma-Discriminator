################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file stores the CNN model used to train on the Melanoma Dataset
# PyTorch's EfficientNet is used, more: https://github.com/lukemelas/EfficientNet-PyTorch#about-efficientnet
# Pooling and Dropout layers are suggested in https://www.kaggle.com/shaitender/melanoma-efficientnet-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import module
import torchvision.models as models

from efficientnet_pytorch import EfficientNet

class EfNet(nn.Module):

    def __init__(self, efnet_version, meta=False, meta_len=24):
        """
        Initializes a model network.
        
        Args:
            efnet_version (int/str): which version to train on [0, 7]
        Vars:
            efnet (EfficientNet): the EfficientNet network to be used
        """
        super(EfNet, self).__init__()
        self.use_meta = meta
        self.efnet_version = efnet_version if (isinstance(efnet_version, int) and \
                                               efnet_version in [i + 1 for i in range(7)]) else Config.EFNET_VER_DEFAULT
        self.efnet = EfficientNet.from_pretrained('efficientnet-b{}'.format(self.efnet_version))
        in_features = getattr(self.efnet, '_fc').in_features

        if self.use_meta:
            self.meta_features = 256
            self.final_fc_features = 128

            self.meta_path = nn.Sequential(
                nn.Linear(meta_len, self.meta_features * 2), \
                nn.ReLU(), \
                nn.Linear(self.meta_features * 2, self.meta_features), \
                nn.ReLU(), \
                nn.Linear(self.meta_features, self.meta_features), \
                nn.ReLU() \
            )
            
            self.final_fc = nn.Sequential(
                nn.Linear(in_features + self.meta_features, self.final_fc_features),
                nn.ReLU(),
                nn.Linear(self.final_fc_features, 1)
            )
        else:
            self.drop = nn.Dropout(0.3)
            self.classifier = nn.Linear(in_features, 1)
    
    def forward(self, x, meta_ensemble=None):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        batch_size = x.shape[0]
        features = self.efnet.extract_features(x)
        if self.use_meta:
            meta_out = self.meta_path(meta_ensemble)
            out = self.final_fc(torch.cat((features, meta_out), dim=1))
        else:
            features = F.adaptive_avg_pool2d(features, 1).reshape(batch_size, -1)
            dropout = self.drop(features)
            out = self.classifier(dropout)
        return out

# Author of the code below: Peng Weiyuan
# parameter settings suggested in https://www.kaggle.com/zzy990106/pytorch-efficientnet-b2-resnext50
class AdaptiveConcatPool2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.max = nn.AdaptiveMaxPool2d(output_size=(1, 1))

    def forward(self, x):
        avg_x = self.avg(x)
        max_x = self.max(x)
        return torch.cat([avg_x, max_x], dim=1)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class ResNeXt(nn.Module):

    def __init__(self, c_out=1, meta=False, meta_len=24):
        super().__init__()
        remove_range = 2
        m = models.resnet18(pretrained=True)

        #c_feature = list(m.children())[-1].in_features
        #self.base = nn.Sequential(*list(m.children())[:-remove_range])
        #self.head = nn.Sequential(
        #    AdaptiveConcatPool2d(),
        #    Flatten(),
        #    nn.Linear(c_feature * 2, c_out)
        #)
        self.use_meta = meta
        self.c_feature = list(m.children())[-1].in_features

        self.base = nn.Sequential(*list(m.children())[:-remove_range])
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(self.c_feature * 2, self.c_feature if self.use_meta else c_out)
        )
        
        if self.use_meta:
            self.meta_features = 256
            self.final_fc_features = 128

            self.meta_path = nn.Sequential(
                nn.Linear(meta_len, self.meta_features * 2), \
                nn.ReLU(), \
                nn.Linear(self.meta_features * 2, self.meta_features), \
                nn.ReLU(), \
                nn.Linear(self.meta_features, self.meta_features), \
                nn.ReLU() \
            )
            
            self.final_fc = nn.Sequential(
                nn.Linear(self.c_feature + self.meta_features, self.final_fc_features),
                nn.ReLU(),
                nn.Linear(self.final_fc_features, c_out)
            )

    def forward(self, x, meta_ensemble=None):
        h = self.base(x)
        logits = self.head(h).squeeze(1)
        if self.use_meta:
            meta_out = self.meta_path(meta_ensemble)
            out = self.final_fc(torch.cat((logits, meta_out), dim=1))
            return out
        else:
            return logits

if __name__ == '__main__':
    net = Net(1)
    