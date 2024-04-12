# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Conv4(torch.nn.Module):
    def __init__(self, exp_dict):
        super().__init__()
        self.conv0 = torch.nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(64)
        self.conv1 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.conv2 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(64)
        self.conv3 = torch.nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.exp_dict = exp_dict
        if self.exp_dict["avgpool"] == True:
            self.output_size = 64
        else:
            self.output_size = 1600
    
    def add_classifier(self, no, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, no))

    def add_adjust(self, ni, no, name="adjust", type='linear', dropout=0.7):
        if type == 'linear':
            setattr(self, name, torch.nn.Linear(ni, no))
        elif type == 'nolinear':
            dim_input =ni
            dim_out = no
            setattr(self, name, nn.Sequential(nn.Linear(dim_input, dim_input),
                                    nn.Dropout(p=dropout),
                                    nn.ReLU(),
                                    nn.Linear(dim_input, dim_out)))
        elif type == 'projection_MLP':
            in_dim, hidden_dim, out_dim = ni, ni, no
            layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            layer2 = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            layer3 = nn.Sequential(
                nn.Linear(hidden_dim, out_dim),
                nn.BatchNorm1d(out_dim)
            )
            setattr(self, name, nn.Sequential(layer1,layer2,layer3))
        elif type == 'prediction_MLP':
            in_dim, hidden_dim, out_dim = ni, ni // 2, no
            layer1 = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True)
            )
            layer2 = nn.Linear(hidden_dim, out_dim)
            setattr(self, name, nn.Sequential(layer1, layer2,))
        else:
            raise NotImplementedError

    def add_sigmoid(self):
        setattr(self, 'sigmoid', nn.Sigmoid())

    def forward(self, x, *args, **kwargs):
        *dim, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.conv0(x) # 84
        x = F.relu(self.bn0(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 84 -> 42
        x = self.conv1(x)
        x = F.relu(self.bn1(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 42 -> 21
        x = self.conv2(x)
        x = F.relu(self.bn2(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 10
        x = self.conv3(x)
        x = F.relu(self.bn3(x), True)
        x = F.max_pool2d(x, 2, 2, 0) # 21 -> 5
        if self.exp_dict["avgpool"] == True:
            x = x.mean(3, keepdim=True).mean(2, keepdim=True)
        return x.view(*dim, self.output_size)