from .version import __version__

def print_banner(case):

    print()
    print("=" * 60)
    print("                     COCKATOO")
    print("          Reactor Physics Computational Framework")
    print("=" * 60)

    print()
    print(f"Version              : {__version__}")

    print()
    print("CASE INFORMATION")
    print("-" * 60)

    print(f"Geometry             : {case.geometry.name}")
    print(f"Mesh                 : {case.mesh.name}")

    print()
    print("PHYSICS")
    print("-" * 60)

    print(f"Physics              : {case.physics}")

    print()
    print("SOLVER")
    print("-" * 60)

    print(f"Solver               : {case.solver}")

    print()
    print("=" * 60)
    print("Starting calculation...")
    print("=" * 60)
    print()