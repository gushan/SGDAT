import copy
import math
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer, required
from torch import Tensor
from typing import List, Optional
import torch.optim._functional as F


class Bop2ndOrder(Optimizer):
    def __init__(
        self, 
        params, 
        gamma: float = 1e-4,
        threshold: float = 1e-8,
        sigma: float = 1e-2,
        name="Bop2ndOrder", 
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
        if sigma < 0:
            raise ValueError(
                'Invalid sigma value: {}'.format(sigma)
            )

        defaults = dict(
            gamma=gamma,
            threshold=threshold,
            sigma=sigma
        )

        super(Bop2ndOrder, self).__init__(params, defaults)

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

                # Bop2ndOrder optimizer
                if hasattr(p,'org'):
                    grad = p.grad.data

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0

                    if not hasattr(p,'m'):
                        p.m = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if not hasattr(p,'v'):
                        p.v = torch.zeros_like(p, memory_format=torch.preserve_format)

                    gamma= group['gamma']
                    threshold = group['threshold']
                    sigma = group['sigma']
                    m = p.m
                    v = p.v
                    state['step'] += 1

                    m_t = m.add_(gamma * (grad - m))
                    v_t = v.add_(sigma * (grad*grad - v))
                    temp = m_t / (v_t.sqrt() + 1e-10)
                    p.data = torch.sign(torch.sign(-torch.sign(p.data.mul(temp) - threshold).mul(p.data)).add(0.1))

        return loss
