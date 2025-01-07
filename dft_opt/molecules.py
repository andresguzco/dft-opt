import numpy as np
from mess.structure import Structure


def get_molecule(name, state):
    mol = MOLECULE_MAP[state][name]
    return mol


MOLECULE_MAP = {
    "Ground": {
        'CH': Structure(
            atomic_number=np.array([6, 1]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        ),
        'O2': Structure(
            atomic_number=np.array([8, 8]),
            position=np.array([[0.0, 0.0, -1], [0.0, 0.0, 1]])
        ),
        'BeH': Structure(
            atomic_number=np.array([4, 1]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])
        ),
        'H2O': Structure(
            atomic_number=np.array([8, 1, 1]),
            position=np.array([
                [0.0, 0.0, 0.0],
                [0.0, 0.76, 0.58],
                [0.0, -0.76, 0.58]
            ])
        ),
        'C4H5N': Structure(
            atomic_number=np.array([6, 6, 7, 6, 6, 1, 1, 1, 1]),
            position=np.array([
                [0.0, 0.0, 0.0],
                [1.4, 0.0, 0.0],
                [2.7, 0.0, 0.0],
                [0.7, 1.2, 0.0],
                [-0.7, 1.2, 0.0],
                [0.7, 2.3, 0.0],
                [-0.7, 2.3, 0.0],
                [2.7, 1.0, 0.0],
                [2.7, -1.0, 0.0]
            ])
        ),
        'C6H8O6': Structure(
            atomic_number=np.array([6, 6, 8, 6, 6, 8, 6, 1, 8]),
            position=np.array([
                [0.0, 0.0, 0.0],
                [1.2, 0.0, 0.0],
                [0.6, 1.2, 0.0],
                [2.4, 0.0, 0.0],
                [3.6, 0.0, 0.0],
                [2.4, 1.2, 0.0],
                [4.8, 0.0, 0.0],
                [5.0, 1.0, 0.0],
                [0.0, -1.2, 0.0]
            ])
        )
    },
    "Excited": {
        'CH': Structure(
            atomic_number=np.array([6, 1]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.1]])
        ),
        'O2': Structure(
            atomic_number=np.array([8, 8]),
            position=np.array([[0.0, 0.0, -0.6], [0.0, 0.0, 0.6]])
        ),
        'BeH': Structure(
            atomic_number=np.array([4, 1]),
            position=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.3]])
        )
    }
}