import time
from pyscf import gto, dft

def get_molecule(molname):
    basis = "6311++g**"
    spin = 0
    if molname == "H2":
        atom_desc = "H -0.5 0 0; H 0.5 0 0"
    elif molname == "Li2":
        atom_desc = "Li -2.5 0 0; Li 2.5 0 0"
    elif molname == "N2":
        atom_desc = "N -1 0 0; N 1 0 0"
    elif molname == "O2":
        atom_desc = "O -1 0 0; O 1 0 0"
        spin = 2
    elif molname == "CO":
        atom_desc = "C -1 0 0; O 1 0 0"
    elif molname == "F2":
        atom_desc = "F -1.25 0 0; F 1.25 0 0"
    else:
        raise RuntimeError("Unknown molecule %s" % molname)

    mol = gto.M(atom=atom_desc, basis=basis, unit="Bohr", spin=spin)
    return mol

def get_atom(atom, spin=0):
    basis = "6311++g**"
    mol = gto.M(atom="%s 0 0 0"%atom, spin=spin, basis=basis, unit="Bohr")
    return mol

def get_molecules_energy(xc="lda", with_df=False):
    molnames = ["H2", "Li2", "N2", "CO", "F2", "O2"]
    for molname in molnames:
        t0 = time.time()
        mol = get_molecule(molname)
        mf = dft.KS(mol)
        if with_df:
            mf = mf.density_fit()
            mf.with_df.auxbasis = "def2-svp-jkfit"
        mf.xc = xc
        mf.grids.level = 4
        energy = mf.kernel()
        t1 = time.time()
        print("Molecule %s: %.8e (%.3e)" % (molname, energy, t1-t0))

def get_atoms_energy():
    atoms = ["H", "Li", "B", "O"]
    spins = [1, 1, 1, 2]
    for (atomname, spin) in zip(atoms, spins):
        t0 = time.time()
        atom = get_atom(atomname, spin)
        mf = dft.UKS(atom)
        mf.xc = "mgga_x_scan"
        mf.grids.level = 4
        energy = mf.kernel()
        t1 = time.time()
        print("Atom %s: %.8e (%.3e)" % (atomname, energy, t1-t0))

if __name__ == "__main__":
    # mol = gto.M(atom="O 0 0 0", basis="6-311++G**", spin=2)
    # mf = dft.UKS(mol)
    # mf.xc = "lda"
    # mf.grids.level = 4
    # print(mf.kernel())
    get_molecules_energy(xc="mgga_x_scan", with_df=False)
    # get_atoms_energy()
