import torch
from torch import nn


class TripletLossIP(nn.Module):
    def __init__(self, margin):
        super(TripletLossIP, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, average=True):
        dist = torch.sum(
                (anchor - positive) ** 2 - (anchor - negative) ** 2,
                dim=1) + self.margin
        dist_hinge = torch.clamp(dist, min=0.0)
        if average:
            return torch.mean(dist_hinge)
        else:
            return dist_hinge
