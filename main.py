####################################
# Imports
####################################
import jax
import time
import argparse
import warnings
from pyscfad import scf
from dft_opt import (
    get_molecule, 
    solve, 
    plot_energy, 
    validate, 
)
####################################

####################################
# Enable 64-bit precision
#########################################
jax.config.update("jax_enable_x64", True)
#########################################


def main():
    ####################################
    # Parse command line arguments
    ####################################
    parser = argparse.ArgumentParser(description="Run DFT optimization benchmarks.")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set to use (e.g., cc-pVDZ)")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name (e.g., H2O)")
    parser.add_argument("--optimizer", type=str, default="BFGS", help="Optimizer to use")
    parser.add_argument("--ortho", type=str, default="qr", help="Orthogonalization method to use")
    parser.add_argument("--num_iter", type=int, default=4000, help="Number of iterations for optimizer")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    ####################################

    ####################################
    # Set up the molecule and basis
    ####################################
    print(f"Parameters: [{args.molecule} / {args.basis} / {args.optimizer}]", flush=True)
    ####################################

    ####################################
    # Benchmark with PySCF
    ####################################
    mol = get_molecule(args.molecule, args.basis)
    mf = scf.RHF(mol)
    mf.kernel()

    start_time = time.time()
    pyscf_energy = mf.kernel()
    pyscf_time = (time.time() - start_time) * 1000
    print(f"PySCF Energy: [{pyscf_energy:.2f}], Time: [{pyscf_time:.2f} ms]", flush=True)
    ####################################
    

    ####################################
    # Solve with JAX
    ####################################
    Z, E, elapsed_time, history = solve(mol, mf, args.num_iter, args.lr, args.optimizer, args.ortho)
    print(f"JAX Energy: [{E:.2f}], Time: [{elapsed_time:.2f} ms]", flush=True)
    plot_energy(history, args)
    ####################################
    
    ####################################
    # Check if the optimized Z is valid
    ####################################
    validate(Z, mol, mf, sum(mol.nelec), args.ortho)
    print("All tests passed!", flush=True)
    ####################################

if __name__ == "__main__":
    main()
