import torch
from torch import eye, zeros, float64, no_grad, outer
from torch.linalg import pinv
from torch.optim.optimizer import Optimizer
from geoopt.tensor import ManifoldParameter, ManifoldTensor

class RBFGS(Optimizer):
    def __init__(self, params, lr=1e-3):
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        assert len(self.param_groups) == 1
        p = self.param_groups[0]["params"][0]

        self.n = p.numel()
        self.H_k = eye(self.n, dtype=float64)
        self.old_p = p.data.view(-1, 1).clone().double()
        self.old_rgrad = zeros(self.n, 1, dtype=float64)

        self.first_step = True

    def step(self, closure):
        loss = closure()

        group = self.param_groups[0]
        lr = group["lr"]
        point = group["params"][0]
        manifold = point.manifold

        with no_grad():
            egrad = point.grad
            rgrad = manifold.egrad2rgrad(point, egrad)

            if self.first_step: 
                eta_k = rgrad
                self.first_step = False
            else:
                eta_k = (self.H_k @ rgrad.view(-1)).view_as(point.data)
                s_k = manifold.tranp(lr * eta_k)
                y_k = rgrad - manifold.transp(self.old_rgrad)

                new_p = manifold.retr(point, -lr * eta_k)
                point.data.copy_(new_p)

                self.old_p = point.data.detach().clone()
                self.old_rgrad = rgrad.detach().clone()

        return loss


 

    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            p.copy_(manifold.projx(p))
