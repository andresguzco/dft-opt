import jax.numpy as jnp
import jax.numpy.linalg as jnl

def cayley(Z, S, X):
    Skew_X = jnp.tril(Z, -1) - jnp.tril(Z, -1).T
    cayley = lambda N: jnl.solve(jnp.eye(N.shape[0]) + N, jnp.eye(N.shape[0]) - N)
    D = cayley(Skew_X)
    D = D @ D
    L_inv_T = jnp.linalg.inv(jnp.linalg.cholesky(S).T)
    C = jnp.matmul(L_inv_T, D)
    return C

def qr(Z, S, X): 
    C = 
    return C

