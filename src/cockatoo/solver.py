class Solution:

    def __init__(self, keff):

        self.keff = keff

    def summary(self):

        print()
        print("-" * 60)
        print("RESULTS")
        print("-" * 60)

        print(f"keff = {self.keff:.6f}")

        print("-" * 60)


class Solver:

    def solve(self, problem):

        print()
        print("Initializing calculation...")
        print(f"Problem type : {problem.problem_type}")

        print("Building geometry...")
        print("Loading materials...")
        print("Assembling equations...")
        print("Solving...")

        # Fake result for now
        keff = 1.023456

        return Solution(keff)