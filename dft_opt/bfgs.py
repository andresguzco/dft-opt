from torch import eye, zeros, float64, no_grad, outer
from torch.linalg import pinv
from torch.optim.optimizer import Optimizer

class BFGS(Optimizer):
    def __init__(self, params, lr=1e-3):
        
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

        assert len(self.param_groups) == 1
        p = self.param_groups[0]["params"][0]

        self.n = p.numel()
        self.H_k = eye(self.n, dtype=float64)
        self.old_p = p.data.view(-1, 1).clone().double()
        self.old_grad = zeros(self.n, 1, dtype=float64)

        self.first_step = True

    def step(self, closure):
        loss = closure()
        group = self.param_groups[0]
        lr = group["lr"]
        p = group["params"][0]

        with no_grad():
            p_data_flat = p.data.flatten().double()
            grad_flat = p.grad.flatten().double()

            if self.first_step:
                p_k = grad_flat
                self.first_step = False
            else:
                y_k = grad_flat.sub(self.old_grad)
                s_k = p_data_flat.sub(self.old_p)

                Hk_s_k = self.H_k @ s_k
                term1 = outer(y_k, y_k) / y_k.dot(s_k)
                term2 = outer(Hk_s_k, Hk_s_k) / s_k.dot(Hk_s_k)

                self.H_k = self.H_k + term1 - term2
                H_inv = pinv(self.H_k)

                p_k = H_inv @ grad_flat

            self.old_p = p_data_flat.clone()
            self.old_grad = grad_flat.clone()

            p_data_flat.add_(p_k.view(p_data_flat.shape), alpha=-lr)
            p.data[:] = p_data_flat.view_as(p.data)

            return loss
    