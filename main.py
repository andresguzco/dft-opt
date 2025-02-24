####################################
# Imports
####################################
import dqc
import time
import argparse
from dft_opt import get_molecule
from pyscf import scf
####################################


######################################
# Optimization and Benchmark fn
######################################
def optimize_energy(basis, structure, ortho_fn, optimizer):
    start_time = time.time()
    m = dqc.Mol(structure, basis=basis, ao_parameterizer=ortho_fn)
    qc = dqc.HF(system=m, variational=True if optimizer=="Adam" else False).run()
    ene = qc.energy()
    return ene, (time.time() - start_time) * 1000
####################################


def main():
    ####################################
    # Parse command line arguments
    ####################################
    parser = argparse.ArgumentParser(description="Run DFT optimization benchmarks.")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set to use (e.g., cc-pVDZ)")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name (e.g., H2O)")
    parser.add_argument("--optimizer", type=str, default="LBFGS", help="Optimizer to use")
    parser.add_argument("--ortho", type=str, default="qr", help="Orthogonalization function to use")
    parser.add_argument("--num_iter", type=int, default=500, help="Number of iterations for optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate for Adam optimizer")
    args = parser.parse_args()
    ####################################

    ####################################
    # Set up the molecule and basis
    ####################################
    print(f"Parameters: [{args.molecule} / {args.basis} / {args.optimizer} / {args.ortho}]", flush=True)
    ####################################

    ####################################
    # Benchmark with PySCF
    ####################################
    mol, structure = get_molecule(args.molecule, args.basis)
    mf = scf.RHF(mol)
    mf.kernel()

    start_time = time.time()
    pyscf_energy = mf.kernel()
    pyscf_time = (time.time() - start_time) * 1000
    print(f"PySCF Energy: [{pyscf_energy:.2f}], Time: [{pyscf_time:.2f} ms]", flush=True)
    ####################################
    
    ####################################
    # Solve with MESS
    ####################################
    E, DQC_time = optimize_energy(
        basis=args.basis, 
        structure=structure, 
        ortho_fn=args.ortho,
        optimizer=args.optimizer
        )
    print(f"DQC Energy: [{E:.2f}], Time: [{DQC_time:.2f} ms]", flush=True)
    ####################################
    
    ####################################
    # Check if the optimized Z is valid
    ####################################
    # validate(Z, H, sum(mol.nelec))
    # print("All tests passed!", flush=True)
    ####################################

if __name__ == "__main__":
    main()
