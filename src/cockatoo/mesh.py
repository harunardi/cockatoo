class Mesh:
    """Base class for numerical meshes."""
    pass


class Cartesian1D(Mesh):
    name = "1D Cartesian"

    def __init__(self, nx, width):

        if nx <= 0:
            raise ValueError(
                "nx must be greater than zero."
            )

        if width <= 0:
            raise ValueError(
                "width must be greater than zero."
            )

        self.nx = nx
        self.width = width
        self.dx = width / nx