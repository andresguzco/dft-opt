import jax
import time
import optax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
from .scf import density_matrix, occupancy, get_X, energy_cayley, energy_qr, qr, cayley


def solve(mol, kernel, iter, lr, optimizer, ortho_fn):
    X = get_X(mol)
    occ = occupancy(mol)
    h1e = kernel.get_hcore()
    e_nuc = kernel.energy_nuc()
    e_disp = kernel.get_dispersion()

    args = (X, kernel, occ, h1e, e_nuc, e_disp)

    n, _ = X.shape

    if optimizer == "BFGS":
        optim = optax.chain(optax.lbfgs(), optax.scale_by_zoom_linesearch(15))
    elif optimizer == "Adam":
        optim = optax.adam(learning_rate=lr)
    else:
        raise ValueError("Invalid optimizer")
    
    energy = energy_cayley if ortho_fn == "cayley" else energy_qr
    
    Z = jnp.eye(n)
    state = optim.init(Z)
    history = jnp.zeros(iter)

    start_time = time.time()
    for i in range(iter):
        e, grads = energy(Z, args)
        updates, state = optim.update(grads, state)
        Z = optax.apply_updates(Z, updates)
        history = history.at[i].set(e)

    elapsed_time = (time.time() - start_time) * 1000
    return Z, e, elapsed_time, history


def validate(Z, mol, kernel, nelec, ortho_fn):
    if ortho_fn == "cayley":
        Q = cayley(Z)
        energy = energy_cayley
    else:
        Q = qr(Z)
        energy = energy_qr
    
    X = get_X(mol)
    occ = occupancy(mol)
    h1e = kernel.get_hcore()
    e_nuc = kernel.energy_nuc()
    disp = kernel.get_dispersion()

    C = X @ Q
    P = density_matrix(C, occ)
    S = jnp.linalg.inv(X @ X.T)

    assert jnp.allclose(jnp.eye(Z.shape[0]), Q.T @ Q, atol=1e-5), "Q is not orthonormal"
    assert jnp.allclose(nelec, jnp.trace(P @ S), atol=1e-5), f"Trace(P @ S) != N"

    Hessian = jax.hessian(lambda l: energy(l, (X, kernel, occ, h1e, e_nuc, disp))[0])(Z)
    assert jnl.eigh(Hessian)[0].all() >= 0, "Hession != PSD"
