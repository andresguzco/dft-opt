import jax
import optax
import jax.numpy as jnp

from tqdm import tqdm
from mess import Hamiltonian, basisset
from mess.structure import nuclear_energy
from dft_opt.plotting import plot_energy
from dft_opt.molecules import get_molecule
from dft_opt.orthonormalize import cayley


def optimize_energy(mol, basis_name, method='pbe', num_iterations=200, learning_rate=0.1):
    basis = basisset(mol, basis_name)
    H = Hamiltonian(basis, xc_method=method)
    optimiser = optax.adam(learning_rate=learning_rate)
    E_n = nuclear_energy(mol)

    @jax.jit
    @jax.value_and_grad
    def total_energy(Z):
        S = H.X.T @ H.X
        C = H.X @ cayley(Z, S)[0]
        # C = H.orthonormalise(Z)
        P = H.basis.density_matrix(C)
        return H(P)

    n = basis.num_orbitals
    Z = jnp.eye(n)
    state = optimiser.init(Z)
    history = []

    for _ in tqdm(range(num_iterations), desc="Optimizing Energy"):
        e, grads = total_energy(Z)
        e += E_n
        updates, state = optimiser.update(grads, state)
        Z = optax.apply_updates(Z, updates)
        history.append(e)
    
    C, _ = cayley(Z, H.X.T @ H.X)
    assert jnp.allclose(C.T @ (H.X.T @ H.X) @ C, jnp.eye(H.basis.num_orbitals))

    # hessian = jax.jacrev(total_energy)(Z)[1]
    # hessian_matrix = hessian.reshape((n * n, n * n))
    # eigenvalues = jnp.linalg.eigvals(hessian_matrix)
    # print(f"Hessian Eigenvalue: {eigenvalues}")
    return jnp.stack(history)


def main():
    mol_name = "O2"
    energy_state = "Ground"
    basis_name = "6-31g"
    
    mol = get_molecule(mol_name, energy_state)
    history = optimize_energy(mol, basis_name)
    plot_energy(history, mol_name, "Cayley")


if __name__ == "__main__":
    main()
