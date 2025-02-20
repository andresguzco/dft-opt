####################################
# Imports
####################################
# import os
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
#
import jax
import time
import argparse
import warnings

from pyscfad import scf, dft
from dft_opt import get_molecule, solve, optimize, plot_energy, validate, Hamiltonian
####################################

####################################
# Enable 64-bit precision
#########################################
jax.config.update("jax_enable_x64", True)
#########################################

######################################
# Optimization and Benchmark functions
######################################
def optimize_energy(H, args):    
    train_fn = solve if args.optimizer == "LBFGS" else  optimize
    train_fn(H, 2, args.lr)
    print(f"Warmup complete!", flush=True)

    start_time = time.time()
    Z, E, history = train_fn(H, args.num_iter, args.lr)
    elapsed_time = (time.time() - start_time) * 1000

    if history is not None:
        plot_energy([val for val in history], args)

    return E , Z, elapsed_time


def main():
    ####################################
    # Parse command line arguments
    ####################################
    parser = argparse.ArgumentParser(description="Run DFT optimization benchmarks.")
    parser.add_argument("--method", type=str, default="hfx", help="Method to use (e.g., hfx, lda)")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set to use (e.g., cc-pVDZ)")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name (e.g., H2O)")
    parser.add_argument("--optimizer", type=str, default="LBFGS", help="Optimizer to use")
    parser.add_argument("--ortho_fn", type=str, default="qr", help="Orthogonalization function to use")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of iterations for optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    ####################################

    ####################################
    # Suppress UserWarnings
    ####################################
    warnings.filterwarnings("ignore", category=UserWarning)
    ####################################

    ####################################
    # Set up the molecule and basis
    ####################################
    print(f"Running computations on: {jax.devices()[0] }", flush=True)
    print(f"Parameters: [{args.molecule} / {args.method} / {args.basis} / {args.optimizer}]", flush=True)
    ####################################

    ####################################
    # Benchmark with PySCF
    ####################################
    mol = get_molecule(args.molecule, args.basis)
    print(f"Molecule built.")
    mf = scf.RHF(mol) if args.method == "hfx" else dft.RKS(mol, xc=args.method)
    print(f"Mean-field initialized.")
    mf.kernel()
    print(f"Mean-field computed.")

    # Benchmark PySCF
    start_time = time.time()
    pyscf_energy = mf.kernel()
    pyscf_time = (time.time() - start_time) * 1000
    print(f"PySCF Energy: [{pyscf_energy:.2f}], Time: [{pyscf_time:.2f} ms]", flush=True)
    ####################################
    
    ####################################
    # Solve with MESS
    ####################################
    H = Hamiltonian(mol=mol, kernel=mf, ortho_fn=0 if args.ortho_fn == 'qr' else 1)
    E, Z, mess_time = optimize_energy(H=H, args=args)
    print(f"MESS Energy: [{E:.2f}], Time: [{mess_time:.2f} ms]", flush=True)
    ####################################
    
    ####################################
    # Check if the optimized Z is valid
    ####################################
    validate(Z, H, sum(mol.nelec))
    print("All tests passed!", flush=True)
    ####################################

if __name__ == "__main__":
    main()
