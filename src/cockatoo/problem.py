class Problem:
    def __init__(self, problem_type, geometry, material,):
        self.problem_type = problem_type
        self.geometry = geometry
        self.material = material

    @classmethod
    def from_dict(cls, data):
        return cls(
            problem_type=data["problem"]["type"],
            geometry=data["geometry"],
            material=data["material"]
        )