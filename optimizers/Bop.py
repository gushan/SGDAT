import copy
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F


class Bop(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        name="Bop", 
        **kwargs
    ):
        if gamma < 0:
            raise ValueError(
                'Invalid gamma value: {}'.format(gamma)
            )
        if threshold < 0:
            raise ValueError(
                'Invalid threshold value: {}'.format(threshold)
            )

        defaults = dict(
            gamma=gamma,
            threshold=threshold
        )

        super(Bop, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                # Bop optimizer
                if hasattr(p,'org'):
                    grad = p.grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0

                    if not hasattr(p,'m'):
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format)

                    gamma= group['gamma']
                    threshold = group['threshold']
                    m = p.m
                    state['step'] += 1

                    p.m.mul_(1-gamma).add_(grad, alpha=gamma)
                    p.data = torch.sign(torch.sign(-torch.sign(p.data.mul(m).add(-threshold)).mul(p.data)).add(0.1))

        return loss
