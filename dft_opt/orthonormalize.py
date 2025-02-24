import jax
import jax.numpy as jnp
import jax.numpy.linalg as jnl


@jax.jit
def qr(Z):
    Q, _ = jnl.qr(Z)
    return Q


@jax.jit
def cayley(Z):
    X = jnp.tril(Z, -1) - jnp.tril(Z, -1).T
    Q = jnl.solve(jnp.eye(X.shape[0]) + X, jnp.eye(X.shape[0]) - X)
    return Q


def validate(Z, H, nelec):
    Q = H.orthonormalize(Z) 
    C = H.X @ Q
    P = H.density_matrix(C)
    S = jnp.linalg.inv(H.X @ H.X.T)

    @jax.jit
    def energy(Z, _):
        C = H.X @ H.orthonormalize(Z)
        P = H.density_matrix(C)
        e = H(P)
        aux = None
        return e, aux

    assert jnp.allclose(jnp.eye(Z.shape[0]), Q.T @ Q, atol=1e-5), "Q is not orthonormal"
    assert jnp.allclose(nelec, jnp.trace(P @ S), atol=1e-5), f"Trace(P @ S) != N"
