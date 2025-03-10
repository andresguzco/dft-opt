import matplotlib.pyplot as plt   
import seaborn as sns

from torch.autograd.functional import hessian
from torch.linalg import qr, solve, inv, eigvalsh
from torch import tril, eye, allclose, trace, einsum, double


def QR(Z):
    Q, _ = qr(Z)
    return Q


def cayley(Z):
    X = tril(Z, -1) - tril(Z, -1).T
    I = eye(X.shape[0])
    Q = solve(I + X, I - X)
    return Q


def density_matrix(C, occupancy):
    return einsum("k,ik,jk->ij", occupancy, C, C)


def plot_energy(history, args):
    sns.set_theme(style="whitegrid")
    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total energy (Hartree)")

    plt.title(f"Energy Optimization of {args.molecule}")

    filename = f"{args.molecule}_{args.basis}_{args.optimizer}_{args.ortho}"
    plt.savefig(f'plots/energy/{filename}.png')
    plt.close()


def validate(kernel, ortho_fn):
    Z, X, P = kernel._Z, kernel._X, kernel._dm, 
    Q = cayley(Z) if ortho_fn == "cayley" else QR(Z)
    C = X @ Q
    S = inv(X @ X.T)

    def energy(z):
        q = cayley(z) if ortho_fn == "cayley" else QR(z)
        c = X @ q
        p = density_matrix(c, kernel._occupancy)
        energy = kernel._engine.dm2energy(p)
        return energy

    I = eye(Z.shape[0], dtype=double)
    n = kernel._nelec

    assert allclose(I, C.T @ S @ C, atol=1e-10), "Q is not orthonormal"
    assert allclose(n, trace(P @ S), atol=1e-10), f"Trace(P @ S) != N"

    Hessian = hessian(lambda X: energy(X), Z)
    eigs = eigvalsh(Hessian)
    assert eigs.all() >= -1e-10, "Hessian != PSD"