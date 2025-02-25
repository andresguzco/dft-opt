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


def validate(Z, H, nelec, ortho_fn):
    if ortho_fn == "cayley":
        Q = cayley(Z)

        @jax.jit
        def energy(Z, X):
            C = X @ cayley(Z)
            P = H.density_matrix(C)
            e = H(P)
            return e
    else:
        Q = qr(Z)

        @jax.jit
        def energy(Z, X):
            C = X @ qr(Z)
            P = H.density_matrix(C)
            e = H(P)
            return e    

    C = H.X @ Q
    P = H.density_matrix(C)
    S = jnp.linalg.inv(H.X @ H.X.T)

    assert jnp.allclose(jnp.eye(Z.shape[0]), Q.T @ Q, atol=1e-5), "Q is not orthonormal"
    assert jnp.allclose(nelec, jnp.trace(P @ S), atol=1e-5), f"Trace(P @ S) != N"

    Hessian = jax.hessian(lambda X: energy(X, H.X))(Z)
    assert jnl.eigh(Hessian)[0].all() >= 0, "Hession != PSD"
