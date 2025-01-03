import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl


def cayley(Z, S):
    Skew_X = jnp.tril(Z, -1) - jnp.tril(Z, -1).T
    cayley = lambda N: jnl.solve(jnp.eye(N.shape[0]) + N, jnp.eye(N.shape[0]) - N)
    D = cayley(Skew_X)
    D = D @ D
    L_inv_T = jnp.linalg.inv(jnp.linalg.cholesky(S).T)
    C = jnp.matmul(L_inv_T, D)
    return C


@jax.jit
@jax.value_and_grad
def energy_qr(Z, H, _):
    C = H.X @ jnl.qr(Z).Q
    P = H.basis.density_matrix(C)
    return H(P)


@jax.jit
@jax.value_and_grad
def energy_cayley(Z, H, S):
    C = H.X @ cayley(Z, S)
    P = H.basis.density_matrix(C)
    return H(P)



