import torch
from pyscf import gto


def get_molecule(name, basis):
    info = PYSCF_MAP[name]
    charge = 1 if name == 'ScCO+' else 0
    mol = gto.M(atom=info, basis=basis, charge=charge)
    mol.verbose = 0
    mol.spin = 0
    atomzs = [val[0] for val in info]
    atompos = torch.cat([torch.tensor(val[1]) for val in info], dim=0).view(-1, 3)
    structure = (atomzs, atompos)
    return mol, structure


PYSCF_MAP = {
        'C6H6': [
                [ 'C', (4.673795, 6.280948, 0.00) ],
                [ 'C', (5.901190, 5.572311, 0.00) ],
                [ 'C', (5.901190, 4.155037, 0.00) ],
                [ 'C', (4.673795, 3.446400, 0.00) ],
                [ 'C', (3.446400, 4.155037, 0.00) ],
                [ 'C', (3.446400, 5.572311, 0.00) ],
                [ 'H', (4.673795, 7.376888, 0.00) ],
                [ 'H', (6.850301, 6.120281, 0.00) ],
                [ 'H', (6.850301, 3.607068, 0.00) ],
                [ 'H', (4.673795, 2.350461, 0.00) ],
                [ 'H', (2.497289, 3.607068, 0.00) ],
                [ 'H', (2.497289, 6.120281, 0.00) ]
                ],
        'H2O': [
                ['O', (0.000000,  0.000000,  0.117266)],
                ['H', (0.000000,  0.755450, -0.469064)],
                ['H', (0.000000, -0.755450, -0.469064)]
                ],
        'ScCO+': [
                ['Sc', (0.00, 0.0, 0.0)],
                ['C' , (1.80, 0.0, 0.0)],
                ['O' , (2.80, 0.0, 0.0)]
                ]
}
