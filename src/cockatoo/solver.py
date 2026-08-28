from .result import EigenvalueResult


class Solver:

    def solve(self, case):

        raise NotImplementedError


class PowerIteration(Solver):

    def __init__(
        self,
        tolerance=1e-8,
        max_iterations=1000,
    ):

        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def solve(self, case):

        print()
        print("Running Power Iteration...")

        # Temporary result
        keff = 1.0

        return EigenvalueResult(
            keff=keff,
            iterations=0,
            converged=False,
        )