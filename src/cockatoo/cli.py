import argparse

from .reader import InputReader
from .run import run


def main():

    parser = argparse.ArgumentParser(
        prog="cockatoo",
        description="Cockatoo Reactor Physics Framework",
    )

    parser.add_argument(
        "input",
        help="Cockatoo input file",
    )

    args = parser.parse_args()

    reader = InputReader(args.input)

    case = reader.read()

    result = run(case)

    result.summary()

    print()
    print("CASE:", case.name)
    print("GEOMETRY:", case.geometry.width)
    print("MESH:", case.mesh.nx)
    print("MATERIAL:", case.materials.get("fuel").name)
    print("PHYSICS:", case.physics.groups)
    print("SOLVER:", type(case.solver).__name__)
    print("TOLERANCE:", case.solver.tolerance)
    print("MAX ITERATIONS:", case.solver.max_iterations)
    print()