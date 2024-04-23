import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, p=2):
        super().__init__()
        self.smooth = smooth
        self.p = p

    def forward(self, pred, tar):
        pred, tar = pred.flatten(1), tar.flatten(1)

        num = (pred*tar).sum(1) + self.smooth
        den = (pred.pow(self.p) + tar.pow(self.p)).sum(1) + self.smooth

        loss = 1 - num/den
        
        return loss.mean()