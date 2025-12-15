import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2., reduction='mean'):
        """
        Args:
            weight (Tensor, optional): A manual rescaling weight given to each class.
                                       If given, has to be a Tensor of size C.
            gamma (float): Focusing parameter.
            reduction (string, optional): Specifies the reduction to apply to the output:
                                          'none' | 'mean' | 'sum'. 'mean': the sum of the output will be divided by the number of elements in the output,
                                          'sum': the output will be summed. Default: 'mean'
        """
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C) where C = number of classes.
            targets: (N) where each value is 0 <= targets[i] <= C-1.
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
