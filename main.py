####################################
# Imports
####################################
import dqc
import time
import torch
import argparse
import csv
import os
import xitorch as xt

from pyscf import scf, dft
from dft_opt import get_molecule, plot_energy, validate
####################################


def main():
    ####################################
    # Parse command line arguments
    ####################################
    parser = argparse.ArgumentParser(description="Run DFT optimization benchmarks.")
    parser.add_argument("--basis", type=str, default="def2-SVP", help="Basis set to use (e.g., cc-pVDZ)")
    parser.add_argument("--molecule", type=str, required=True, help="Molecule name (e.g., H2O)")
    parser.add_argument("--optimizer", type=str, default="bfgs", help="Optimizer to use")
    parser.add_argument("--ortho", type=str, default="qr", help="Orthogonalization function to use")
    parser.add_argument("--iters", type=int, default=1000, help="Number of iterations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--benchmark", action="store_true", help="Run PySCF benchmark")    
    parser.add_argument("--disable_tex", action="store_true", default=False, help="Disable TeX rendering in matplotlib.")
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
    if args.benchmark:
        # mf = scf.RHF(mol)
        mf = dft.UKS(mol, xc="B3LYP")

        start_time = time.time()
        pyscf_energy = mf.kernel()
        pyscf_time = (time.time() - start_time) * 1000

        print(f"PySCF Energy: [{pyscf_energy:.2f}], Time: [{pyscf_time:.2f} ms]", flush=True)
    ####################################
    

    ####################################
    # Solve with JAX
    ####################################
    m = dqc.Mol(structure, basis=args.basis, ao_parameterizer=args.ortho)

    # basis = m._atombases
    # bpacker = xt.Packer(basis)  # use xitorch's Packer to get the tensors within a structure
    # bparams = bpacker.get_param_tensor()  # get the parameters of the basis as one tensor
    # print(bparams.shape)
    # raise ValueError

    # qc = dqc.HF(system=m)
    qc = dqc.KS(system=m, xc="HYB_GGA_XC_B3LYP")

    start_time = time.time()
    qc.run(opt_type=args.optimizer, iter=args.iters, lr=args.lr)
    time_elapsed = (time.time() - start_time) * 1000

    print(f"DQC Energy: [{qc.energy():.2f}], Time: [{time_elapsed:.2f} ms]", flush=True)
    ####################################
    filename = f"data/{args.molecule}_{args.basis}_{args.optimizer}.csv"

    if os.path.exists(filename):
        with open(filename, 'r', newline='') as csvfile:
            reader = list(csv.reader(csvfile))
        
        reader[0].append(f"{args.ortho}")
        for i, row in enumerate(reader[1:-1]):
            row.append(str(qc.history[i].item()))
        
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(reader)
    else:
        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([f"{args.ortho}"])
            for i, val in enumerate(qc.history):
                writer.writerow([val.item()])

    with open("data/timings.csv", mode='a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([args.molecule, args.basis, args. optimizer, args.ortho, time_elapsed])

    plot_energy(args, pyscf_energy)
    ####################################
    
    ####################################
    # Check if the optimized Z is valid
    ####################################
    validate(qc, args.ortho)
    print("All tests passed!", flush=True)
    ####################################

if __name__ == "__main__":
    main()
