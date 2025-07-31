# newly written
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

import pdb

@MODELS.register_module()
class EvidentialMSELoss(nn.Module):
    """"EvidentialMSELoss.
    
    """

    def __init__(self,
                 loss_weight = 1.0,
                 kl_strength = 1.0,
                 reduction='mean',
                 loss_name='loss_evidential',
                 ):
        super().__init__()
        self.loss_weight = loss_weight
        self.kl_strength = kl_strength
        self.reduction = reduction
        self._loss_name = loss_name
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None, # TODO ?
                reduction="mean",
                ignore_index=255,
                **kwargs):
        """Forward function."""
        # pdb.set_trace()

        evidence = F.softplus(pred)
        alpha = evidence + 1
        S = torch.sum(alpha, dim = 1, keepdim = True) # TODO dim?
        probs = alpha / S
        # uc_map = probs.shape[1] / (S + 1)

        # ignore index
        new_target = target.clone()
        new_target[new_target == ignore_index] = 0

        target_onehot = F.one_hot(new_target, num_classes = pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float()

        valid_mask = (target != ignore_index).unsqueeze(1).float()
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        mse = torch.sum((target_onehot - probs)**2, dim = 1)

        kl = self.kl_strength * self._kl_dirichlet(alpha)

        loss = mse + kl

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        
        return self.loss_weight * loss
    
    def _kl_dirichlet(self, alpha):
        # pdb.set_trace()
        """KL divergence to uniform Dirichlet (1,...,1)"""
        K = alpha.shape[1]
        beta = torch.ones_like(alpha)

        sum_alpha = torch.sum(alpha, dim=1, keepdim = True)
        sum_beta  = torch.sum(beta,  dim=1, keepdim = True)
        
        lnB_alpha = torch.sum(torch.lgamma(alpha), dim=1) - torch.lgamma(sum_alpha.squeeze(1))
        lnB_beta  = torch.sum(torch.lgamma(beta),  dim=1) - torch.lgamma(sum_beta.squeeze(1))

        digamma_alpha = torch.digamma(alpha)
        digamma_sum_alpha = torch.digamma(sum_alpha)

        kl = torch.sum((alpha - beta) * (digamma_alpha - digamma_sum_alpha), dim = 1) + lnB_alpha - lnB_beta

        return kl
    
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name


