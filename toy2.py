import torch
import geoopt as go
import matplotlib.pyplot as plt

torch.manual_seed(0)

n = 4
steps = 200

A = torch.randn(n, n, dtype=torch.float64)
B = torch.randn(n, n, dtype=torch.float64)

def loss_fn(C):
    """
    loss(C) = - ||A C - B||^2_F
    """
    diff = C @ A - B
    return - diff.norm(p="fro") ** 2

manifold = go.manifolds.Stiefel()
C_init = torch.randn(n, n, dtype=torch.float64)
C_init = manifold.projx(C_init)
C = go.ManifoldParameter(C_init, manifold=manifold)

optimizer = go.optim.RiemannianAdam([C], lr=0.1)

history = torch.zeros(steps, dtype=torch.float64)
for step in range(steps):
    optimizer.zero_grad()
    loss = loss_fn(C)
    loss.backward()
    optimizer.step()
    history[step] = -loss.item()

with torch.no_grad():
    assert torch.allclose(C.T @ C, torch.eye(n, dtype=torch.float64), atol=1e-5), "C^T C != I"

plt.style.use("ggplot")
plt.figure()
plt.plot(history.numpy())
plt.xlabel("Iteration")
plt.ylabel(r"$f(C) = \|C A - B\|_{F}^{2}$")
plt.title("Maximizing ||C A - B||² on the Stiefel manifold")
plt.savefig(f'plots/toy2.png')
plt.close()
