import jax
import optax
import numpy as np
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tqdm import tqdm
from mess import Hamiltonian, basisset
from pyquante2 import molecule
from mess.structure import Structure
from mess.structure import nuclear_energy


MOLECULE_MAP = {
    "Ground": {
        'CH': molecule(
            [
                (6, 0.0, 0.0, 0.0), 
                (1, 0.0, 0.0, 1.1)
            ],
            units='Angstrom',
            # multiplicity=2,
            name='Methane'
        ),
        'O2': molecule(
            [
                (8, 0.0, 0.0, -0.6), 
                (8, 0.0, 0.0, 0.6)
            ],
            units='Angstrom',
            # multiplicity=3,
            name='Oxygen'
        ),
        'BeH': molecule(
            [
                (4, 0.0, 0.0, 0.0), 
                (1, 0.0, 0.0, 1.3)
            ],
            units='Angstrom',
            # multiplicity=2,
            name='Beryllium Hydride'
        )
    },
    "Excited": {
        'CH': molecule(
            [
                (6, 0.0, 0.0, 0.0),
                (1, 0.0, 0.0, 1.1)
            ],
            units='Angstrom',
            # multiplicity=1,  # Singlet state for CH+
            charge=1,
            name='Methane (Cationic Excited)'
        ),
        'O2': molecule(
            [
                (8, 0.0, 0.0, -0.6),
                (8, 0.0, 0.0, 0.6)
            ],
            units='Angstrom',
            # multiplicity=2,  # Doublet state for O2+
            charge=1,
            name='Oxygen (Cationic Excited)'
        ),
        'BeH': molecule(
            [
                (4, 0.0, 0.0, 0.0),
                (1, 0.0, 0.0, 1.3)
            ],
            units='Angstrom',
            # multiplicity=1,  # Singlet state for BeH+
            charge=1,
            name='Beryllium Hydride (Cationic Excited)'
        )
    }
}

def get_molecule(name, state):
    mol = MOLECULE_MAP[state][name]
    atomic_number, position = zip(*[(a.Z, a.r) for a in mol])
    atomic_number, position = [np.asarray(x) for x in (atomic_number, position)]
    return Structure(atomic_number, position)


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

    hessian = jax.jacrev(total_energy)(Z)[1]
    hessian_matrix = hessian.reshape((n * n, n * n))
    eigenvalues = jnp.linalg.eigvals(hessian_matrix)
    print(f"Hessian Eigenvalue: {eigenvalues}")
    return jnp.stack(history)


def main():
    sns.set_theme(style="whitegrid")
    mol_name = "O2"
    energy_state = "Ground"
    basis_name = "6-31g"

    mol = get_molecule(mol_name, energy_state)
    history = optimize_energy(mol, basis_name)

    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total energy (Hartree)")
    plt.title("Energy Optimization of CH")
    plt.savefig('plots/energy.png')
    plt.close()


if __name__ == "__main__":
    main()