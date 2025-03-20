import torch
import geoopt as go
import matplotlib.pyplot as plt

torch.manual_seed(0)

n, k = 4, 2
steps = 200

M = torch.randn(n, n, dtype=torch.float64)
A = 0.5 * (M + M.T) 
A = A + n * torch.eye(n)

def loss_fn(C):
    """
    loss(C) = trace(C^T A C)
    """
    return torch.trace(C.T @ A @ C)

manifold = go.manifolds.Stiefel()  

C_init = torch.randn(n, k, dtype=torch.float64)
C_init = manifold.projx(C_init)
C = go.ManifoldParameter(C_init, manifold=manifold)

optimizer = go.optim.RiemannianAdam([C], lr=0.1)

history = torch.zeros(steps)
for step in range(200):
    optimizer.zero_grad()
    f = loss_fn(C)
    f.backward()
    optimizer.step()
    history[step] = f.item()

with torch.no_grad():
    assert torch.allclose(C.T @ C, torch.eye(k, dtype=torch.float64), atol=1e-5), "C^T C != I"

plt.style.use("ggplot")
plt.figure()
plt.plot(data=history)
plt.xlabel("Iteration")
plt.ylabel("Total energy (Hartree)")

plt.title(f"Loss = trace(C^T A C) on Stiefel manifold")
plt.savefig(f'plots/toy.png')
plt.close()