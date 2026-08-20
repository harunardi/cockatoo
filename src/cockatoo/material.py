class Material:

    def __init__(
        self,
        name,
        D,
        Sigma_a,
        nuSigma_f,
    ):

        self.name = name
        self.D = D
        self.Sigma_a = Sigma_a
        self.nuSigma_f = nuSigma_f


class MaterialCollection:

    def __init__(self):

        self._materials = {}

    def add(self, material):

        if material.name in self._materials:
            raise ValueError(
                f"Material '{material.name}' already exists."
            )

        self._materials[material.name] = material

    def get(self, name):

        return self._materials[name]

    def __len__(self):

        return len(self._materials)