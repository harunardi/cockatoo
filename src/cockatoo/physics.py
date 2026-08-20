class Physics:
    """Base class for physical models."""
    pass


class Diffusion(Physics):

    def __init__(self, groups=1):

        if groups <= 0:
            raise ValueError(
                "Number of groups must be positive."
            )

        self.groups = groups