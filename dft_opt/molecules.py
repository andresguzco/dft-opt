
import numpy as np
from pyquante2 import molecule
from mess.structure import Structure


def get_molecule(name, state):
    mol = MOLECULE_MAP[state][name]
    atomic_number, position = zip(*[(a.Z, a.r) for a in mol])
    atomic_number, position = [np.asarray(x) for x in (atomic_number, position)]
    return Structure(atomic_number, position)


MOLECULE_MAP = {
    "Ground": {
        'CH': molecule(
            [
                (6, 0.0, 0.0, 0.0), 
                (1, 0.0, 0.0, 1.1)
            ],
            units='Angstrom',
            name='Methane'
        ),
        'O2': molecule(
            [
                (8, 0.0, 0.0, -0.6), 
                (8, 0.0, 0.0, 0.6)
            ],
            units='Angstrom',
            name='Oxygen'
        ),
        'BeH': molecule(
            [
                (4, 0.0, 0.0, 0.0), 
                (1, 0.0, 0.0, 1.3)
            ],
            units='Angstrom',
            name='Beryllium Hydride'
        ),
        'H2O': molecule(
            [
                (8, 0.0, 0.0, 0.0),
                (1, 0.0, 0.76, 0.58),
                (1, 0.0, -0.76, 0.58)
            ],
            units='Angstrom',
            name='Water'
        ),
        'C4H5N': molecule(
            [
                (6, 0.0, 0.0, 0.0), 
                (6, 1.4, 0.0, 0.0), 
                (7, 2.7, 0.0, 0.0), 
                (6, 0.7, 1.2, 0.0),
                (6, -0.7, 1.2, 0.0),
                (1, 0.7, 2.3, 0.0),
                (1, -0.7, 2.3, 0.0),
                (1, 2.7, 1.0, 0.0),
                (1, 2.7, -1.0, 0.0)
            ],
            units='Angstrom',
            name='Pyrrole'
        ),
        'C6H8O6': molecule(
            [
                (6, 0.0, 0.0, 0.0),
                (6, 1.2, 0.0, 0.0),
                (8, 0.6, 1.2, 0.0),
                (6, 2.4, 0.0, 0.0),
                (6, 3.6, 0.0, 0.0),
                (8, 2.4, 1.2, 0.0),
                (6, 4.8, 0.0, 0.0),
                (1, 5.0, 1.0, 0.0),
                (8, 0.0, -1.2, 0.0)
            ],
            units='Angstrom',
            name='Ascorbic Acid (Vitamin C)'
        )
    },
    "Excited": {
        'CH': molecule(
            [
                (6, 0.0, 0.0, 0.0),
                (1, 0.0, 0.0, 1.1)
            ],
            units='Angstrom',
            charge=1,
            name='Methane (Cationic Excited)'
        ),
        'O2': molecule(
            [
                (8, 0.0, 0.0, -0.6),
                (8, 0.0, 0.0, 0.6)
            ],
            units='Angstrom',
            charge=1,
            name='Oxygen (Cationic Excited)'
        ),
        'BeH': molecule(
            [
                (4, 0.0, 0.0, 0.0),
                (1, 0.0, 0.0, 1.3)
            ],
            units='Angstrom',
            charge=1,
            name='Beryllium Hydride (Cationic Excited)'
        )
    }
}