from .material import MaterialCollection

_active_case = None


class Case:

    def __init__(self):

        global _active_case

        self.name = "unnamed_case"
        self.geometry = None
        self.materials = MaterialCollection()
        self.mesh = None
        self.physics = None
        self.solver = None

        _active_case = self

def get_active_case():

    return _active_case