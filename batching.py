import jax
import optax
import numpy as np
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt

from tqdm import tqdm 
from mess import Hamiltonian, basisset
from mess.structure import Structure, nuclear_energy


def h2_hamiltonian(r: float, basis_name: str = "sto-3g", xc_method="lda"):
    mol = Structure(
        atomic_number=np.array([1, 1]),
        position=np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]]),
    )
    basis = basisset(mol, basis_name=basis_name)
    return Hamiltonian(basis, xc_method=xc_method)

@jax.jit
@jax.vmap
def energy(Z, H):
    C = H.orthonormalise(Z)
    P = H.basis.density_matrix(C)
    return H(P)

@jax.value_and_grad
def loss_fn(z, h):
    return jnp.sum(energy(z, h))


def optimize_h2(rs, num_iterations=128, learning_rate=0.1):
    num_confs = len(rs)
    H = [h2_hamiltonian(r) for r in rs]
    num_orbitals = H[0].basis.num_orbitals
    H = jax.tree.map(lambda *xs: jnp.stack(xs), *H)
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

    return history, rs, E_total


def plot_results(history, rs, E_total):
    sns.set_theme(style="whitegrid")

    plt.figure()
    ax = sns.lineplot(data=history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Batched Loss (Hartree)")
    plt.title("Optimization Loss History")
    plt.savefig('plots/loss.png')
    plt.close()

    plt.figure()
    ax = sns.lineplot(x=rs, y=E_total)
    ax.set_xlabel("$H_2$ bond length (a.u.)")
    ax.set_ylabel("Total Energy (Hartree)")
    plt.title("H2 Bond Length vs Total Energy")
    plt.savefig('plots/distance.png')
    plt.close()


def main():
    num_confs = 64
    rs = np.linspace(0.6, 60, num_confs)
    history, rs, E_total = optimize_h2(rs)
    plot_results(history, rs, E_total)


if __name__ == "__main__":
    main()