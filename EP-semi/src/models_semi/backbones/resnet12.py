# -*- coding: utf-8 -*-
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn

class Block(torch.nn.Module):
    def __init__(self, ni, no, stride, dropout=0, groups=1):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(dropout) if dropout > 0 else lambda x: x
        self.conv0 = torch.nn.Conv2d(ni, no, 3, stride, padding=1, bias=False)
        self.bn0 = torch.nn.BatchNorm2d(no)
        self.conv1 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn1 = torch.nn.BatchNorm2d(no)
        self.conv2 = torch.nn.Conv2d(no, no, 3, 1, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(no)
        if stride == 2 or ni != no:
            self.shortcut = torch.nn.Conv2d(ni, no, 1, stride=1, padding=0)

    def get_parameters(self):
        return self.parameters()

    def forward(self, x, is_support=True):
        y = F.relu(self.bn0(self.conv0(x)), True)
        y = self.dropout(y)
        y = F.relu(self.bn1(self.conv1(y)), True)
        y = self.dropout(y)
        y = self.bn2(self.conv2(y))
        return F.relu(y + self.shortcut(x), True)


class Resnet12(torch.nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.output_size = 512
        assert(width == 1) # Comment for different variants of this model
        self.widths = [x * int(width) for x in [64, 128, 256]]
        self.widths.append(self.output_size * width)
        self.bn_out = torch.nn.BatchNorm1d(self.output_size)

        start_width = 3
        for i in range(len(self.widths)):
            setattr(self, "group_%d" %i, Block(start_width, self.widths[i], 1, dropout))
            start_width = self.widths[i]

    def add_classifier(self, nclasses, name="classifier", modalities=None):
        setattr(self, name, torch.nn.Linear(self.output_size, nclasses))

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

    def up_to_embedding(self, x, is_support):
        """ Applies the four residual groups
        Args:
            x: input images
            n: number of few-shot classes
            k: number of images per few-shot class
            is_support: whether the input is the support set (for non-transductive)
        """
        for i in range(len(self.widths)):
            x = getattr(self, "group_%d" % i)(x, is_support)
            x = F.max_pool2d(x, 3, 2, 1)
        return x

    def forward(self, x, is_support=True):
        """Main Pytorch forward function

        Returns: class logits

        Args:
            x: input mages
            is_support: whether the input is the sample set
        """
        *args, c, h, w = x.size()
        x = x.view(-1, c, h, w)
        x = self.up_to_embedding(x, is_support)
        return F.relu(self.bn_out(x.mean(3).mean(2)), True)