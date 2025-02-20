import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl


@jax.jit
def cayley(Z):
    X = jnp.tril(Z, -1) - jnp.tril(Z, -1).T
    Q = jnl.solve(jnp.eye(X.shape[0]) + X, jnp.eye(X.shape[0]) - X)
    return Q


@jax.jit
def qr(Z):
    Q, _ = jnl.qr(Z)
    return Q
