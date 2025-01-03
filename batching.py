import jax
import optax
import numpy as np
import jax.numpy as jnp

from tqdm import tqdm 
from mess import Hamiltonian, basisset
from mess.structure import Structure, nuclear_energy
from dft_opt.plotting import plot_batched_results


@jax.jit
@jax.vmap
def energy(Z, H):
    C = H.orthonormalise(Z)
    P = H.basis.density_matrix(C)
    return H(P)

@jax.value_and_grad
def loss_fn(z, h):
    return jnp.sum(energy(z, h))


def optimize(Hs, num_iterations=128, learning_rate=0.1):
    num_confs = len(Hs)
    num_orbitals = Hs[0].basis.num_orbitals

    H = jax.tree.map(lambda *xs: jnp.stack(xs), *Hs)
    Z = jnp.tile(jnp.eye(num_orbitals), (num_confs, 1, 1))

    optimiser = optax.adam(learning_rate=learning_rate)
    state = optimiser.init(Z)

    history = []
    for _ in tqdm(range(num_iterations), desc="Optimizing H2"):
        loss, grads = loss_fn(Z, H)
        updates, state = optimiser.update(grads, state)
        Z = optax.apply_updates(Z, updates)
        history.append(loss)

    history = jnp.stack(history)
    E_n = jax.vmap(nuclear_energy)(H.basis.structure)
    E_total = energy(Z, H) + E_n

    return history, E_total


def main():
    basis_name: str = "sto-3g"
    xc_method: str ="lda"
    num_confs = 1
    rs = np.linspace(2, 2, num_confs)

    mols = [Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
    ) for r in rs]
    
    bases = [basisset(mol, basis_name=basis_name) for mol in mols]
    Hamiltonians = [Hamiltonian(basis, xc_method=xc_method) for basis in bases]

    history, E_total = optimize(Hamiltonians)
    plot_batched_results(history, rs, E_total)


if __name__ == "__main__":
    main()