import time
import optax
import jax.numpy as jnp
import jax.numpy.linalg as jnl
import optimistix as optx
import numpy as np
import pandas as pd
import warnings

from mess import Hamiltonian, basisset
from mess.structure import nuclear_energy
from pyscf import dft, scf
from mess.interop import to_pyscf
from dft_opt.molecules import get_molecule
from dft_opt.loss import energy_qr, energy_cayley, cayley


def optimize_energy(
        method, 
        basis, 
        E_n,
        ortho_fn, 
        optimizer,
        num_iterations=200, 
        learning_rate=0.1
        ):
    
    H = Hamiltonian(basis, xc_method=method)
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
        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000 
        return E_total, sol.value, elapsed_time
    
    elif optimizer == "Adam":
        counter = 0
        history = []
        optimiser = optax.adam(learning_rate=learning_rate)
        state = optimiser.init(Z)

        for _ in range(num_iterations):
            e, grads = loss_fn(Z, H, S)
            e += E_n
            updates, state = optimiser.update(grads, state)
            Z = optax.apply_updates(Z, updates)
            history.append(e)

            if len(history)> 1 and (history[-1] - history[-2]) / history[-2] < 1e-3:
                counter += 1

            if counter > 5:
                break

        end_time = time.time()
        elapsed_time = (end_time - start_time) * 1000 
        return jnp.stack(history)[-1], Z, elapsed_time
    
    else:
        raise ValueError("Invalid optimizer")
    

def timeit_multiple_runs(func, num_runs=5, warm_up=True, *args, **kwargs):
    if warm_up:
        print("Performing warm-up...")
        func(*args, **kwargs)  # Warm-up run

    times = []
    result = None
    for _ in range(num_runs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = (time.time() - start_time) * 1000
        times.append(elapsed_time)
    avg_time = np.mean(times)
    return avg_time, result


def mess_benchmark(method, basis, E_n, ortho_fn, optimizer, num_runs=5):
    avg_time, res = timeit_multiple_runs(
        optimize_energy, num_runs=num_runs, warm_up=True,
        method=method, basis=basis, E_n=E_n, ortho_fn=ortho_fn, optimizer=optimizer
    )
    return avg_time, res


def pyscf_benchmark(mol, basis_name, method, num_runs=5, warm_up=False):
    scf_mol = to_pyscf(mol, basis_name)
    scf_mol.verbose = 0

    def pyscf_run():
        if method == "hfx":
            scf_instance = scf.RHF(scf_mol)
        else:
            scf_instance = dft.RKS(scf_mol, xc=method)
        return scf_instance.kernel()

    avg_time, pyscf_energy = timeit_multiple_runs(pyscf_run, num_runs=num_runs, warm_up=warm_up)
    return avg_time, pyscf_energy


def benchmark_methods(mol_names, energy_state, basis_name, methods, ortho_fns, optimizers, num_runs=5, validate=False):
    time_results = {}
    energy_results = {}

    for mol_name in mol_names:
        mol = get_molecule(mol_name, energy_state)
        scf_mol = to_pyscf(mol, basis_name)

        basis = basisset(mol, basis_name)
        E_n = nuclear_energy(mol)
        print(f"\n=== Benchmarking for Molecule: {mol_name.upper()} with n={basis.num_orbitals} ===")

        for method in methods:
            H = Hamiltonian(basis, xc_method=method)
            identifier = f"{mol_name} {method} / {basis_name}"
            time_results[identifier] = {}
            energy_results[identifier] = {}

            pyscf_time, pyscf_energy = pyscf_benchmark(mol, basis_name, method, num_runs)
            time_results[identifier]["PySCF"] = pyscf_time
            energy_results[identifier]["PySCF"] = pyscf_energy

            print(f"PySCF {method.upper()} Execution Time: {pyscf_time:.2f} ms")
            print(f"PySCF Energy: {pyscf_energy:.2f}")

            for optimizer in optimizers:
                for ortho_fn in ortho_fns:
                    try:
                        mess_time, mess_result = mess_benchmark(method, basis, E_n, ortho_fn, optimizer, num_runs)
                        mess_energy = mess_result[0]
                        mess_column = f"MESS_{optimizer}_{ortho_fn}"
                        time_results[identifier][mess_column] = mess_time
                        energy_results[identifier][mess_column] = mess_energy

                        print(f"MESS {method.upper()} Execution Time ({optimizer}, {ortho_fn}): {mess_time:.2f} ms")
                        print(f"MESS Energy: {mess_energy:.2f}")
                        
                        if validate:
                            if ortho_fn == "cayley":
                                C = H.X @ cayley(mess_result[1], H.X.T @ H.X)
                                P = H.basis.density_matrix(C)
                            else:
                                C = H.X @ jnl.qr(mess_result[1]).Q
                                P = H.basis.density_matrix(C)

                            if method == "hfx":
                                mf = scf.RHF(mol=scf_mol).density_fit()
                            else:
                                mf = dft.RKS(scf_mol, xc=method).density_fit()

                            mf.verbose = 0  

                            print(f"P shape: {P.shape}")

                            hcore = mf.get_hcore(scf_mol)
                            veff = mf.get_veff(scf_mol, P)
                            total_energy = mf.energy_tot(P, hcore, veff)

                            print(f"PySCF Energy with MESS Density: {total_energy:.2f}") 
                    
                    except Exception as e:
                        raise ValueError(f"Problem with ({method, optimizer, ortho_fn})")

    time_df = pd.DataFrame.from_dict(time_results, orient="index")
    energy_df = pd.DataFrame.from_dict(energy_results, orient="index")

    print(time_df.round(2))
    print(energy_df.round(2))

def main():
    # TODO: Add proper check for the density matrix. DONE
    # TODO: Check the orthogonal matrix with PySCF. DONE
    # TODO: Check Wu's Riemannian Adam paper on Slack
    # TODO: Look at the convergence rate of the energy. Look at early stopping with relative change. DONE

    warnings.filterwarnings("ignore")

    mol_names = ["H2O", "C4H5N", "C6H8O6"]
    energy_state = "Ground"
    basis_name = "cc-pVDZ"
    methods = ["lda", "hfx"]
    ortho_fns = ["qr", "cayley"]
    optimizers = ["LBFGS", "Adam"]

    benchmark_methods(mol_names, energy_state, basis_name, methods, ortho_fns, optimizers)

if __name__ == "__main__":
    main()