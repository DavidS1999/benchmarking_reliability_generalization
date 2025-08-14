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
                 ignore_index = 255,
                 kl_anneal_max_i = 10000, # increase kl strength linear
                 min_kl_factor = 0.
                 ):
        super().__init__()
        self.loss_weight = loss_weight
        self.kl_strength = kl_strength
        self.reduction = reduction
        self._loss_name = loss_name
        self.ignore_index = ignore_index
        self.kl_anneal_max_i = kl_anneal_max_i
        self.min_kl_factor = min_kl_factor

        # iter counter
        self.register_buffer("seen_iters", torch.tensor(0, dtype=torch.long), persistent=False)
    
    def _anneal_factor(self):
        if self.training and self.kl_anneal_max_i is not None and self.kl_anneal_max_i > 0:
            self.seen_iters += 1
            factor = max(0, min(1, self.seen_iters.float() / float(self.kl_anneal_max_i)))
        else:
            factor = 1.0
        return self.min_kl_factor + (1 - self.min_kl_factor) * factor
    
    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None, # TODO ?
                reduction=None,
                ignore_index=None,
                **kwargs):
        """Forward function."""
        import pdb
        pdb.set_trace()

        reduction = self.reduction if reduction is None else reduction
        ignore_index = self.ignore_index if ignore_index is None else ignore_index

        evidence = F.softplus(pred)   # evidence >= 0
        alpha = evidence + 1          # alpha >= 1 (if alpha < 1 -> bimodal dirichlet)
        S = torch.sum(alpha, dim = 1, keepdim = True) # TODO dim?
        probs = alpha / S # dim of alpha is [1, 19, 1024, 1024]
        # uc_map = probs.shape[1] / (S + 1)

        # ignore index
        new_target = target.clone()
        new_target[new_target == ignore_index] = 0

        target_onehot = F.one_hot(new_target, num_classes = pred.shape[1])
        target_onehot = target_onehot.permute(0, 3, 1, 2).float() # shape[1, 19, 1024, 1024]

        valid_mask = (target != ignore_index).unsqueeze(1).float()
        probs = probs * valid_mask
        target_onehot = target_onehot * valid_mask

        # mse = torch.sum((target_onehot - probs)**2, dim = 1)
        # kl = self.kl_strength * self._kl_dirichlet(alpha)
        # loss = mse + kl

        # include variance
        mse = (target_onehot - probs) ** 2 # bias
        var = (probs * (1.0 - probs)) / (S + 1.0)
        data_term = (mse + var).sum(dim = 1)

        kl = self._kl_dirichlet(alpha)

        # annealing
        kl_factor = self._anneal_factor() * self.kl_strength

        loss = data_term + kl_factor * kl


        if weight is not None:
            loss = loss*weight # TODO check dimensions

        if reduction == "none":
            loss = loss
        if reduction == "mean":
            valid = valid_mask[:, 0, ...]
            denom = valid.sum().clamp_min(1.0) # don't divide by zero
            loss = (loss*valid).sum() / denom
        elif reduction == "sum":
            loss = loss.sum()
        
        return self.loss_weight * loss
    
    def _kl_dirichlet(self, alpha):
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

    def extra_repr(self):
        return f'kl_strength={self.kl_strength}, loss_weight={self.loss_weight}, reduction={self.reduction}, ignore_index={self.ignore_index}, loss_name="{self._loss_name}"'

