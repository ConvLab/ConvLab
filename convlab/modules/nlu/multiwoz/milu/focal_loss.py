# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalBCEWithLogitsLoss(nn.Module):
    def __init__(self, gamma=0, size_average=True):
        super(FocalBCEWithLogitsLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, input, target):
        logpt = F.logsigmoid(input)
        pt = torch.sigmoid(input)
        loss = -((1-pt)**self.gamma * logpt * target + pt**self.gamma * (1-pt).clamp(min=1e-8).log() * (1-target))

        if self.size_average: 
            return loss.mean()
        return loss.sum()