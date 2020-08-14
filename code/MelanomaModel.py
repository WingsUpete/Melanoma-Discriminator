################## Melanoma Discrimator #####################
###        Created by Peter S on Aug 11th, 2020           ###
###                petershen815@126.com                   ###
### Data Source: https://challenge2020.isic-archive.com/  ###
#############################################################
# This file stores the CNN model used to train on the Melanoma Dataset
# PyTorch's EfficientNet is used, more: https://github.com/lukemelas/EfficientNet-PyTorch#about-efficientnet

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.efnet = EfficientNet.from_pretrained('efficientnet-b{}'.format(self.efnet_version), num_classes=1)
    
    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        out = self.efnet(x)
        return out

if __name__ == '__main__':
    net = Net(2)
    