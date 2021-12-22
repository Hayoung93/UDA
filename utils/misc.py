import torch
from torch import nn


class SharpenSoftmax(nn.Module):
    def __init__(self, tau, dim=0):
        super().__init__()
        self.tau = tau
        self.dim = dim
    
    def forward(self, pred):
        pred = pred / self.tau
        return pred.log_softmax(self.dim)


def get_tsa_mask(pred, max_epoch, epoch, iter_per_epoch, iteration):
    # Use linear TSA strategy
    max_iter = max_epoch * iter_per_epoch
    tsa_th = (epoch * iter_per_epoch + iteration + 1) / max_iter
    return pred.softmax(dim=1) <= tsa_th