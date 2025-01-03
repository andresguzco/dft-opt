import time
import jax
import optax
import jax.numpy as jnp
import optimistix as optx
import numpy as np
import pandas as pd
import warnings

from tqdm import tqdm
from mess import Hamiltonian, basisset
from mess.structure import nuclear_energy
from pyscf import dft, scf
from mess.interop import to_pyscf
from dft_opt.plotting import plot_energy
from dft_opt.molecules import get_molecule
from dft_opt.loss import energy_qr, energy_cayley


def optimize_energy(
        mol, 
        basis_name, 
        method, 
        ortho_fn, 
        optimizer,
        num_iterations=200, 
        learning_rate=0.1
        ):
    
    basis = basisset(mol, basis_name)
    H = Hamiltonian(basis, xc_method=method)
    E_n = nuclear_energy(mol)
    n = basis.num_orbitals
    Z = jnp.eye(n)  

    if ortho_fn == "cayley":
        S = H.X.T @ H.X
        loss_fn = energy_cayley
    else:
        S = jnp.eye(n)
        loss_fn = energy_qr

    start_time = time.time() 

    if optimizer == "LBFGS":
        solver = optx.BFGS(atol=1e-6, rtol=1e-5)
        sol = optx.minimise(lambda Z, _: loss_fn(Z, H, S)[0], solver, Z, max_steps=2000)
        E_total = loss_fn(sol.value, H, S)[0] + E_n
        end_time = time.time()  # End timing
        elapsed_time = (end_time - start_time) * 1000 
        # print(f"LBFGS Optimization Time: {elapsed_time:.3f} ms")
        return E_total, sol, elapsed_time
    
    elif optimizer == "Adam":
        history = []
        optimiser = optax.adam(learning_rate=learning_rate)
        state = optimiser.init(Z)

        # for _ in tqdm(range(num_iterations), desc="Optimizing Energy"):
        for _ in range(num_iterations):
            e, grads = loss_fn(Z, H, S)
            e += E_n
            updates, state = optimiser.update(grads, state)
            Z = optax.apply_updates(Z, updates)
            history.append(e)

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000 
        # print(f"Adam Optimization Time: {elapsed_time:.3f} ms")
        return jnp.stack(history)[-1], None, elapsed_time
    
    else:
        raise ValueError("Invalid optimizer")
    

def timeit_multiple_runs(func, num_runs=5, *args, **kwargs):
    times = []
    result = None
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000 
        times.append(elapsed_time)
    avg_time = np.mean(times)
    return avg_time, result


def mess_benchmark(mol, basis_name, method, ortho_fn, optimizer, num_runs=5):
    avg_time, res = timeit_multiple_runs(
        optimize_energy, num_runs, mol, basis_name, method, ortho_fn, optimizer
    )
    return avg_time, res


def pyscf_benchmark(mol, basis_name, method, num_runs=5):
    scf_mol = to_pyscf(mol, basis_name)
    scf_mol.verbose = 0

    def pyscf_run():
        if method == "hfx":
            scf_instance = scf.RHF(scf_mol)
        else:
            scf_instance = dft.RKS(scf_mol, xc=method)
        return scf_instance.kernel()

    avg_time, pyscf_energy = timeit_multiple_runs(pyscf_run, num_runs)
    return avg_time, pyscf_energy


def benchmark_methods(mol_names, energy_state, basis_name, methods, ortho_fns, optimizers, num_runs=5):
    results = []

    for mol_name in mol_names:
        print(f"\n=== Benchmarking for Molecule: {mol_name.upper()} ===")
        mol = get_molecule(mol_name, energy_state)

        for optimizer in optimizers:
            for ortho_fn in ortho_fns:
                for method in methods:
                    try:
                        identifier = f"{mol_name} {method} / {basis_name}"

                        mess_time, mess_result = mess_benchmark(mol, basis_name, method, ortho_fn, optimizer, num_runs)
                        mess_energy = mess_result[0]
                        column_name = f"MESS_{optimizer}_{ortho_fn}"

                        pyscf_time, pyscf_energy = pyscf_benchmark(mol, basis_name, method, num_runs)

                        results.append({
                            "Identifier": identifier,
                            column_name: mess_time,
                            f"PySCF_{method}": pyscf_time,
                            f"MESS_Energy_{optimizer}_{ortho_fn}": mess_energy,
                            f"PySCF_Energy_{method}": pyscf_energy,
                        })

                        print(f"MESS {method.upper()} Execution Time ({optimizer}, {ortho_fn}): {mess_time:.3f} ms")
                        print(f"MESS Energy: {mess_energy:.6f}")
                        print(f"PySCF {method.upper()} Execution Time: {pyscf_time:.3f} ms")
                        print(f"PySCF Energy: {pyscf_energy:.6f}")
                    except Exception as e:
                        print(f"Error benchmarking {method} for {mol_name}: {e}")

    df = pd.DataFrame(results)
    print(df)

def main():
    warnings.filterwarnings("ignore")

    mol_name = ["H2O", "C4H5N", "C6H8O6"]
    energy_state = "Ground"
    basis_name = "cc-pVDZ"
    methods = ["lda", "hfx"]
    ortho_fns = ["qr", "cayley"]
    optimizers = ["Adam", "LBFGS"]

    benchmark_methods(mol_name, energy_state, basis_name, methods, ortho_fns, optimizers)


if __name__ == "__main__":
    main()