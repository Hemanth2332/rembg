from torch import nn
import torch

class LossFn(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def dice_loss(self, pred, target, smooth=1e-6):
        pred = torch.sigmoid(pred)

        pred = pred.view(pred.size(0), -1)
        target = target.view(target.size(0), -1)

        intersection = (pred * target).sum(dim=1)
        dice = (2. * intersection + smooth) / (
            pred.sum(dim=1) + target.sum(dim=1) + smooth
        )

        return 1 - dice.mean()
    
    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice_loss(pred, target)