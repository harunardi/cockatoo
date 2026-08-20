import numpy as np

class Field:

    def __init__(self, mesh, groups=1):

        self.mesh = mesh
        self.groups = groups

        self.data = np.zeros(
            (groups, mesh.nx),
            dtype=float,
        )