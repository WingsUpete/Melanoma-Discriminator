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

    def __init__(self):
        """
        Initializes a model network.
        
        Vars:
            efnet (EfficientNet): the EfficientNet network to be used
        """
        super(Net, self).__init__()
        self.efnet = EfficientNet.from_pretrained('efficientnet-b1')
        #ef_last_fc_in_features = self.efnet._fc.in_features     # 1280 for b1
        ef_last_fc_out_features = self.efnet._fc.in_features        # 1000 for b1
        #self.efnet._fc = nn.Linear(in_features=ef_last_fc_in_features, out_features=ef_last_fc_out_features, bias=True)
        self.midproc = nn.Linear(ef_last_fc_out_features, 500)
        self.output = nn.Linear(500, 1)
    
    def forward(self, x):
        """
        Performs forward pass of the input.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        """
        z = self.efnet(x)
        z = self.midproc(z)
        out = self.output(z)
        return out

if __name__ == '__main__':
    net = Net()
    