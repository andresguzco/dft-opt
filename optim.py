import jax
import optax
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tqdm import tqdm
from mess import Hamiltonian, basisset, molecule
from mess.interop import from_pyquante
from mess.structure import nuclear_energy


def optimize_energy(mol, basis_name, method='pbe', num_iterations=200, learning_rate=0.1):
    basis = basisset(mol, basis_name)
    H = Hamiltonian(basis, xc_method=method)
    optimiser = optax.adam(learning_rate=learning_rate)

    E_n = nuclear_energy(mol)

    @jax.jit
    @jax.value_and_grad
    def total_energy(Z):
        C = H.orthonormalise(Z)
        P = H.basis.density_matrix(C)
        return H(P)

    Z = jnp.eye(basis.num_orbitals)
    state = optimiser.init(Z)
    history = []

    for _ in tqdm(range(num_iterations), desc="Optimizing Energy"):
        e, grads = total_energy(Z)
        e+= E_n
        updates, state = optimiser.update(grads, state)
        Z = optax.apply_updates(Z, updates)
        history.append(e)


    hessian_fn = jax.hessian(total_energy)
    hessian = hessian_fn(Z)
    print(hessian)

    return jnp.stack(history)


def main():
    sns.set_theme(style="whitegrid")

    mol = from_pyquante("ch4")
    # mol = molecule("water")
    basis_name = "6-31g"

    history = optimize_energy(mol, basis_name)

    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total energy (Hartree)")
    plt.title("Energy Optimization of CH4")
    plt.savefig('plots/energy.png')
    plt.close()


if __name__ == "__main__":
    main()