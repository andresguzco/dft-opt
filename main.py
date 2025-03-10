####################################
# Imports
####################################
import dqc
import time
import torch
import argparse

from pyscf import scf
from dft_opt import get_molecule, plot_energy, validate
####################################


def main():
    ####################################
    # Parse command line arguments
    ####################################
    parser = argparse.ArgumentParser(description="Run DFT optimization benchmarks.")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set to use (e.g., cc-pVDZ)")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name (e.g., H2O)")
    parser.add_argument("--xc", type=str, default="lda_x", help="Exchange-correlation functional to use")
    parser.add_argument("--optimizer", type=str, default="lbfgs", help="Optimizer to use")
    parser.add_argument("--ortho", type=str, default="qr", help="Orthogonalization function to use")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    print(f"Parameters: [{args.molecule} / {args.basis} / {args.optimizer} / {args.ortho} / {args.seed}]", flush=True)
    ####################################

    ####################################
    # Set random seed
    ####################################
    torch.manual_seed(args.seed)
    ####################################

    ####################################
    # Benchmark with PySCF
    ####################################
    mol, structure = get_molecule(args.molecule, args.basis)
    mf = scf.RHF(mol)

    start_time = time.time()
    pyscf_energy = mf.kernel()
    pyscf_time = (time.time() - start_time) * 1000

    print(f"PySCF Energy: [{pyscf_energy:.2f}], Time: [{pyscf_time:.2f} ms]", flush=True)
    ####################################
    
    ####################################
    # Solve with MESS
    ####################################
    m = dqc.Mol(structure, basis=args.basis, ao_parameterizer=args.ortho)
    qc = dqc.HF(system=m)

    start_time = time.time()
    qc.run(opt_type=args.optimizer, iter=args.iters)
    time_elapsed = (time.time() - start_time) * 1000

    print(f"DQC Energy: [{qc.energy():.2f}], Time: [{time_elapsed:.2f} ms]", flush=True)
    plot_energy(qc.history, args)
    ####################################
    
    ####################################
    # Check if the optimized Z is valid
    ####################################
    validate(qc, args.ortho)
    print("All tests passed!", flush=True)
    ####################################

if __name__ == "__main__":
    main()
