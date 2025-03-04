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
    I = jnp.eye(X.shape[0])
    Q = jnl.solve(I + X, I - X)
    return Q


@jax.jit
@jax.value_and_grad
def energy_cayley(Z, args):
    X, kernel, occ, h1e, e_nuc, disp = args
    C = X @ cayley(Z)
    P = density_matrix(C, occ)

    vhf = kernel.get_veff(dm=P)

    e1 = jnp.einsum('ij,ji->', h1e, P).real
    e_coul = jnp.einsum('ij,ji->', vhf, P).real * .5

    e_tot = e1 + e_coul + e_nuc + disp
    return e_tot


@jax.jit
@jax.value_and_grad
def energy_qr(Z, args):
    X, kernel, occ, h1e, e_nuc, disp = args
    C = X @ qr(Z)
    P = density_matrix(C, occ)
    vhf = kernel.get_veff(dm=P)

    e1 = jnp.einsum('ij,ji->', h1e, P).real
    e_coul = jnp.einsum('ij,ji->', vhf, P).real * .5

    e_tot = e1 + e_coul + e_nuc + disp
    return e_tot


@jax.jit
def get_X(mol):
    S = jnp.array(mol.intor(f"int1e_ovlp"))
    N = 1 / jnp.sqrt(jnp.diagonal(S))
    overlap = N[:, jnp.newaxis] * N[jnp.newaxis, :] * S
    s, U = jnl.eigh(overlap)
    s = jnp.diag(jnp.power(s, -0.5))
    X = U @ s @ U.T
    return X


@jax.jit
def occupancy(mol) -> jnp.ndarray:
    occ = jnp.full(mol.nao, 2.0)
    mask = occ.cumsum() > mol.tot_electrons()
    occ = jnp.where(mask, 0.0, occ)
    return occ


@jax.jit
def density_matrix(C: jnp.ndarray, occupancy: jnp.ndarray) ->jnp.ndarray:
    return jnp.einsum("k,ik,jk->ij", occupancy, C, C)



