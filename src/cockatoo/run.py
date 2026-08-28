from .case import get_active_case


def run(case=None):

    if case is None:
        case = get_active_case()

    if case is None:
        raise RuntimeError(
            "No Case has been defined."
        )

    if case.solver is None:
        raise RuntimeError(
            "No solver has been defined."
        )

    return case.solver.solve(case)