import torch
from pyscf import gto


def get_molecule(name, basis):
    info = PYSCF_MAP[name]
    charge = 0 if name == "Fe(CO)2+" else 1 if '+' in name else 0
    mol = gto.M(atom=info, basis=basis, charge=charge)
    mol.verbose = 0
    mol.spin = 2 if name == "Fe(CO)2+" else 0
    atomzs = [val[0] for val in info]
    atompos = torch.cat([torch.tensor(val[1]) for val in info], dim=0).view(-1, 3)
    structure = (atomzs, atompos)
    return mol, structure


PYSCF_MAP = {
    'H2O': [
        ['O', (0.000000,  0.000000,  0.117266)],
        ['H', (0.000000,  0.755450, -0.469064)],
        ['H', (0.000000, -0.755450, -0.469064)]
    ],
    'CH': [
        ['C', (0.000, 0.000, 0.000)],
        ['H', (1.100, 0.000, 0.000)]
    ],
    'OH': [
        ['O', (0.000, 0.000, 0.000)],
        ['H', (0.970, 0.000, 0.000)]
    ],
    'NiCH2+': [
        ['Ni', (0.000,  0.000, 0.000)],
        ['C' , (1.800,  0.000, 0.000)],
        ['H' , (2.400,  0.800, 0.000)],
        ['H' , (2.400, -0.800, 0.000)]
    ],
    'CoCO+': [
        ['Co', (0.000, 0.000, 0.000)],
        ['C' , (1.800, 0.000, 0.000)],
        ['O' , (2.800, 0.000, 0.000)]
    ],
    'NiCO+': [
        ['Ni', (0.000, 0.000, 0.000)],
        ['C' , (1.800, 0.000, 0.000)],
        ['O' , (2.800, 0.000, 0.000)]
    ],
    'ScCO+': [
        ['Sc', (0.00, 0.0, 0.0)],
        ['C' , (1.80, 0.0, 0.0)],
        ['O' , (2.80, 0.0, 0.0)]
    ],
    'Fe(CO)2+': [
        ['Fe', (0.000,   0.000,  0.000)],
        ['C' , (1.800,   0.000,  0.000)],
        ['O' , (2.800,   0.000,  0.000)],
        ['C' , (-1.800,  0.000,  0.000)],
        ['O' , (-2.800,  0.000,  0.000)]
    ]
}

