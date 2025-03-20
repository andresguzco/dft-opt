import pandas as pd
import matplotlib.pyplot as plt
from torch.linalg import qr, solve, inv, svd
from torch import tril, eye, allclose, trace, einsum, double, matrix_exp


def QR(Z):
    Q, _ = qr(Z)
    return Q


def cayley(Z):
    X = tril(Z, -1) - tril(Z, -1).T
    I = eye(X.shape[0])
    Q = solve(I + X, I - X)
    return Q


def matexp(Z):
    A = tril(Z, -1)
    X = A - A.T
    Q = matrix_exp(X)
    return Q


def polar(m):
    U, _, Vh = svd (m)
    u = U @ Vh
    # p = Vh.T.conj() @ S.diag().to (dtype = m.dtype) @ Vh
    return  u


def density_matrix(C, occupancy):
    return einsum("k,ik,jk->ij", occupancy, C, C)


def plot_energy(args):
    df = pd.read_csv(f'data/{args.molecule}_{args.basis}_{args.optimizer}.csv')

    grouped_cols = {}
    for col in df.columns:
        base_name = col.split('.')[0]
        grouped_cols.setdefault(base_name, []).append(col)

    iterations = df.index

    plt.figure(figsize=(10, 6))
    for base_name, cols in grouped_cols.items():
        if len(cols) > 1:
            mean_vals = df[cols].mean(axis=1)
            std_vals = df[cols].std(axis=1)
        else:
            mean_vals = df[cols[0]]
            std_vals = 0
        
        plt.plot(iterations, mean_vals, label=base_name)

        if not isinstance(std_vals, int):  
            plt.fill_between(
                iterations, 
                mean_vals - std_vals, 
                mean_vals + std_vals, 
                alpha=0.2
            )

    plt.xlabel('Iteration')
    plt.ylabel('Energy per Iteration')
    plt.title('Average Performance with Error Bars')
    plt.legend()
    plt.tight_layout()

    filename = f"{args.molecule}_{args.basis}_{args.optimizer}"
    plt.savefig(f'plots/{filename}.png')
    plt.close()


def validate(kernel, ortho_fn):
    Z, X, P = kernel._Z, kernel._X, kernel._dm 
    S = inv(X @ X.T)
    I = eye(Z.shape[-1], dtype=double)

    if ortho_fn == "cayley":
        fn = cayley
    elif ortho_fn == "qr":
        fn = QR
    elif ortho_fn == "polar":
        fn = polar
    elif ortho_fn == "matexp":
        fn = matexp
    else:
        raise ValueError(f"Unknown orthogonalization function: {ortho_fn}")
    
    if Z.dim() != 3:
        Q = fn(Z)
        C = X @ Q
        assert allclose(I, C.T @ S @ C, atol=1e-10), "Q is not orthonormal"
    else:
        Q1, Q2 = fn(Z[0, :, :]), fn(Z[1, :, :]) 
        C1, C2 = X @ Q1, X @ Q2
        assert allclose(I, C1.T @ S @ C1, atol=1e-10), "Q is not orthonormal"
        assert allclose(I, C2.T @ S @ C2, atol=1e-10), "Q is not orthonormal"

    assert allclose(kernel._nelec, trace(P @ S), atol=1e-10), f"Trace(P @ S) != N"