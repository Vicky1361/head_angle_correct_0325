from torch import nn
from torch.nn import functional as F
import torch

def dice_coef(input, target):
    smooth = 1.
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()

    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
#loss使用0.5*交叉熵loss + 0.5*dice loss
class my_loss(nn.Module):
    def __init__(self):
        super(my_loss, self).__init__()
    def forward(self, y_pred, y_true):
        return 0.5 * (1 - dice_coef(y_pred, y_true)) + F.binary_cross_entropy(y_pred, y_true) * 0.5
