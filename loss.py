# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 
# @Author  : Elliott Zheng  
# @Email   : admin@hypercube.top

# Source: https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py

import torch
from torch import nn

# torch.log  and math.log is e based
class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, foreground_weight=10):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.foreground_weight = foreground_weight

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
    
        # Weight map - heavily penalize errors on keypoint pixels
        weight_map = torch.ones_like(y)
        weight_map[y >= self.theta] = self.foreground_weight
    
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        w1 = weight_map[delta_y < self.theta]
        w2 = weight_map[delta_y >= self.theta]
    
        loss1 = w1 * self.omega * torch.log(
            1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * \
            (self.alpha - y2) * (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * \
            (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = w2 * (A * delta_y2 - C)
    
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))