import math
import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor, threshold
from typing import List, Optional


class SGDAT(Optimizer):

    def __init__(self, params, lr=1e-4, threshold=1e-6, weight_decay = 0, momentum = 0, nesterov=False, dampening=0, eps=0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, threshold=threshold, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov, dampening=dampening, eps=eps)
        super(SGDAT, self).__init__(params, defaults)

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
                if hasattr(p,'org'):
                    if p.grad is not None:
                        grad = p.grad.data
                        
                        state = self.state[p]

                        if len(state) == 0:
                            state['flip_num'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                        if not hasattr(p,'m'):
                            p.m = torch.zeros_like(p, memory_format=torch.preserve_format)

                        lr=group['lr']
                        threshold=group['threshold']
                        weight_decay=group['weight_decay']
                        eps=group['eps']
                        dampening=group['dampening']
                        momentum=group['momentum']
                        nesterov = group['nesterov']
                        flip_num = state['flip_num']
                        exp_avg = state['exp_avg']
                        d_p = grad
                       

                        if weight_decay != 0:
                            d_p = d_p.add(p, alpha=weight_decay)

                        if momentum != 0:
                            exp_avg.mul_(momentum).add_(grad, alpha=1 - dampening)

                            if nesterov:
                                d_p = d_p.add(exp_avg, alpha=momentum)
                            else:
                                d_p = exp_avg

                        p.m.add_(d_p, alpha=-lr)    

                        temp = threshold*flip_num
                        p.data = torch.sign(torch.sign(torch.where(p.m.abs()>temp, p.m, p.pre_binary_data)).add(0.1)) 
                        flip_num.add_(torch.ne(torch.sign(p.data),p.pre_binary_data)) 
        return loss
