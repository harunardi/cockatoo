class EigenvalueResult:

    def __init__(
        self,
        keff,
        iterations,
        converged,
    ):

        self.keff = keff
        self.iterations = iterations
        self.converged = converged

    def summary(self):

        print()
        print("=" * 60)
        print("COCKATOO RESULT")
        print("=" * 60)

        print(f"keff       : {self.keff:.8f}")
        print(f"iterations : {self.iterations}")
        print(f"converged  : {self.converged}")

        print("=" * 60)