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

class Net(nn.Module):

    def __init__(self, efnet_version):
        """
        Initializes a model network.
        
        Args:
            efnet_version (int/str): which version to train on [0, 7]
        Vars:
            efnet (EfficientNet): the EfficientNet network to be used
        """
        super(Net, self).__init__()
        self.efnet_version = efnet_version if (isinstance(efnet_version, int) and \
                                               efnet_version in [i + 1 for i in range(7)]) else Config.EFNET_VER_DEFAULT
        self.efnet = EfficientNet.from_pretrained('efficientnet-b{}'.format(self.efnet_version))
        in_features = getattr(self.efnet, '_fc').in_features
        self.drop = nn.Dropout(0.3)
        self.classifier = nn.Linear(in_features, 1)
    
    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        batch_size = x.shape[0]
        features = self.efnet.extract_features(x)
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

    def __init__(self, c_out=1):
        super().__init__()
        remove_range = 2
        m = models.resnet18(pretrained=True)
        # m = models.resnext50_32x4d(pretrained=True)
        # m = torch.hub.load('pytorch/vision:v0.6.0',
        #                    'resnext50_32x4d', pretrained=True)

        c_feature = list(m.children())[-1].in_features
        self.base = nn.Sequential(*list(m.children())[:-remove_range])
        self.head = nn.Sequential(
            AdaptiveConcatPool2d(),
            Flatten(),
            nn.Linear(c_feature * 2, c_out)
        )
        # self.base = nn.Sequential(*list(m.children())[:-remove_range])
        # self.head = nn.Sequential(
        #     AdaptiveConcatPool2d(),
        #     Flatten(),
        #     nn.Linear(c_feature * 2, c_out)
        # )
        

    def forward(self, x):
        h = self.base(x)
        logits = self.head(h).squeeze(1)
        return logits

if __name__ == '__main__':
    net = Net(1)
    