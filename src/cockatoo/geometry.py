class Geometry:
    """Base class for all Cockatoo geometries."""
    pass


class Slab(Geometry):
    name = "1D Slab"

    def __init__(self, width):

        self.width = width