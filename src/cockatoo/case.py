import yaml

from .problem import Problem
from .solver import Solver

class Case:
    def __init__(self, problem: Problem, solver: Solver):
        self.problem = problem
        self.solver = solver

    @classmethod
    def from_file(cls, file_path: str):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        problem = Problem.from_dict(data)
        solver = Solver()

        return cls(problem, solver)

    def run(self):
        # Solve the problem using the solver
        solution = self.solver.solve(self.problem)
        return solution