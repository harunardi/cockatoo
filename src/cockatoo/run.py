from .case import get_active_case
from .banner import print_banner

def run(case=None):

    if case is None:
        case = get_active_case()

    if case.solver is None:
        raise RuntimeError(
            "No solver has been defined."
        )

    print_banner(case)

    return case.solver.solve(case)