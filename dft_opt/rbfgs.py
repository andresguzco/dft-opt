import torch
from torch import eye, zeros, float64, no_grad, outer
from torch.linalg import inv
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
        p = group["params"][0]
        manifold = p.manifold

        with no_grad():
            rgrad = manifold.egrad2rgrad(p, p.grad)

            if self.first_step:
                direction = rgrad.view(-1)
                self.first_step = False
            else:
                transp_old_rgrad = manifold.transp(p, rgrad, self.old_rgrad)
                y_k = rgrad.reshape(-1) - transp_old_rgrad.reshape(-1)

                p_data_flat = p.data.reshape(-1).double()
                s_k = p_data_flat - self.old_p.reshape(-1)

                Hs = self.H_k @ s_k
                term1 = outer(y_k, y_k) / y_k.dot(s_k)
                term2 = outer(Hs, Hs) / s_k.dot(Hs)
                self.H_k.add_(term1).sub(term2)

                H_inv = inv(self.H_k)
                direction = H_inv @ rgrad.view(-1)

            direction_reshaped = direction.view_as(p.data)
            new_p = manifold.retr(p, -lr * direction_reshaped)
            p.data.copy_(new_p)

            self.old_p = p.data.clone()
            self.old_rgrad = rgrad.clone()

            print

        return loss


    @torch.no_grad()
    def stabilize_group(self, group):
        for p in group["params"]:
            if not isinstance(p, (ManifoldParameter, ManifoldTensor)):
                continue
            manifold = p.manifold
            p.copy_(manifold.projx(p))
