# -*- coding: utf-8 -*-
# @Time    : 2019/9/9 
# @Author  : Elliott Zheng  
# @Email   : admin@hypercube.top
# Source: https://github.com/elliottzheng/AdaptiveWingLoss/blob/master/adaptive_wing_loss.py
# Modified: Fixed foreground/background normalization and weight balancing
import torch
from torch import nn

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1, foreground_weight=50):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.foreground_weight = foreground_weight

    def forward(self, pred, target):
        y     = target
        y_hat = pred
        delta_y = (y - y_hat).abs()

        # Weight map based on TARGET value (where ground truth peaks are)
        foreground_mask = y >= self.theta
        weight_map = torch.ones_like(y)
        weight_map[foreground_mask] = self.foreground_weight

        # Split by error magnitude (AWL formulation)
        small_error_mask = delta_y < self.theta
        large_error_mask = ~small_error_mask

        delta_y1 = delta_y[small_error_mask]
        delta_y2 = delta_y[large_error_mask]
        y1       = y[small_error_mask]
        y2       = y[large_error_mask]
        w1       = weight_map[small_error_mask]
        w2       = weight_map[large_error_mask]

        loss1 = w1 * self.omega * torch.log(
            1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))

        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * \
            (self.alpha - y2) * \
            (torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * \
            (1 / self.epsilon)
        C = self.theta * A - self.omega * torch.log(
            1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = w2 * (A * delta_y2 - C)

        # Normalize foreground and background separately so the
        # ~200:1 bg:fg pixel ratio doesn't drown out peak supervision
        fg_mask_small = foreground_mask[small_error_mask]
        fg_mask_large = foreground_mask[large_error_mask]

        fg_loss = (loss1[fg_mask_small].sum() + loss2[fg_mask_large].sum()) / \
                   foreground_mask.sum().clamp(min=1).float()

        bg_loss = (loss1[~fg_mask_small].sum() + loss2[~fg_mask_large].sum()) / \
                   (~foreground_mask).sum().clamp(min=1).float()

        return fg_loss + bg_loss