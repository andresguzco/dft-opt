import time
import numpy as np
import warnings

from mess import molecule, basisset, minimise, Hamiltonian
from pyscf import dft, scf
from mess.interop import to_pyscf


def mess_benchmark(H):
    start_time = time.time()
    _, _, _ = minimise(H)
    end_time = time.time()
    return end_time - start_time


def pyscf_benchmark(scf_instance):
    start_time = time.time()
    scf_instance.kernel()
    end_time = time.time()
    return end_time - start_time


def main():
    mol = molecule("water")
    basis_name = "6-31g"
    basis = basisset(mol, basis_name)
    scf_mol = to_pyscf(mol, basis_name)
    scf_mol.verbose = 0

    H = Hamiltonian(basis, xc_method="pbe")
    E, _, _ = minimise(H)
    s = dft.RKS(scf_mol, xc="pbe")
    s.kernel()
    assert np.allclose(s.energy_tot(), E), "Energies do not match!"

    mess_time = mess_benchmark(H)
    print(f"MESS Minimize Execution Time: {mess_time:.3f} seconds")
    pyscf_time = pyscf_benchmark(s)
    print(f"PySCF Kernel Execution Time: {pyscf_time:.3f} seconds")
    print(f"Time Ratio (PySCF/MESS): {pyscf_time / mess_time:.3f}")

    H_hfx = Hamiltonian(basis, xc_method="hfx")
    E_hfx, _, _ = minimise(H_hfx)
    s_rhf = scf.RHF(scf_mol)
    s_rhf.kernel()
    assert np.allclose(s_rhf.energy_tot(), E_hfx), "Energies do not match!"

    hfx_time = mess_benchmark(H_hfx)
    print(f"HFX Minimize Execution Time: {hfx_time:.3f} seconds")
    pyscf_rhf_time = pyscf_benchmark(s_rhf)
    print(f"PySCF RHF Kernel Execution Time: {pyscf_rhf_time:.3f} seconds")
    print(f"Time Ratio (PySCF/MESS): {pyscf_rhf_time / hfx_time:.3f}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()