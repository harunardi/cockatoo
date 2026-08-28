from .case import Case
from .material import Material
from .geometry import Slab
from .mesh import Cartesian1D
from .physics import Diffusion
from .solver import PowerIteration


class InputReader:

    def __init__(self, filename):

        self.filename = filename
        self.sections = {}
        self.input_version = None

    def read(self):

        # Read file
        with open(self.filename, "r") as f:
            lines = f.readlines()

        # Parse input
        self._parse(lines)

        # Create case
        case = Case()

        # Read individual sections
        self._read_case(case)
        self._read_geometry(case)
        self._read_mesh(case)
        self._read_material(case)
        self._read_physics(case)
        self._read_solver(case)

        return case

    # ======================================================
    # PARSER
    # ======================================================

    def _parse(self, lines):

        current_section = None

        for line_number, line in enumerate(lines, start=1):

            line = line.strip()

            # Empty line
            if not line:
                continue

            # Comment
            if line.startswith("#"):
                continue

            # Input version
            if line.startswith("COCKATOO_INPUT_VERSION"):

                _, value = line.split("=", 1)

                self.input_version = int(
                    value.strip()
                )

                continue

            # Section
            if line.startswith("[") and line.endswith("]"):

                current_section = line[1:-1].strip()

                if current_section in self.sections:
                    raise ValueError(
                        f"Duplicate section "
                        f"'{current_section}' "
                        f"at line {line_number}"
                    )

                self.sections[current_section] = {}

                continue

            # Key-value
            if "=" in line:

                if current_section is None:
                    raise ValueError(
                        f"Parameter outside a section "
                        f"at line {line_number}: {line}"
                    )

                key, value = line.split("=", 1)

                key = key.strip()
                value = value.strip()

                self.sections[current_section][key] = value

                continue

            raise ValueError(
                f"Invalid syntax at line "
                f"{line_number}: {line}"
            )

        # Validate input version
        if self.input_version is None:

            raise ValueError(
                "Missing COCKATOO_INPUT_VERSION."
            )

        if self.input_version != 1:

            raise ValueError(
                f"Unsupported Cockatoo input version: "
                f"{self.input_version}"
            )

    # ======================================================
    # CASE
    # ======================================================

    def _read_case(self, case):

        section = self.sections.get("CASE")

        if section is None:
            return

        case.name = section.get(
            "name",
            "unnamed_case"
        )

    # ======================================================
    # GEOMETRY
    # ======================================================

    def _read_geometry(self, case):

        section = self.sections.get("GEOMETRY")

        if section is None:
            return

        geometry_type = section.get("type")

        if geometry_type == "slab":

            width = float(
                section["width"]
            )

            case.geometry = Slab(
                width=width
            )

        else:

            raise ValueError(
                f"Unknown geometry type: "
                f"{geometry_type}"
            )

    # ======================================================
    # MESH
    # ======================================================

    def _read_mesh(self, case):

        section = self.sections.get("MESH")

        if section is None:
            return

        mesh_type = section.get("type")

        if mesh_type == "cartesian":

            dimension = int(
                section.get("dimension", 1)
            )

            if dimension != 1:
                raise NotImplementedError(
                    "Only 1D Cartesian mesh "
                    "is currently supported."
                )

            nx = int(section["cells"])

            if case.geometry is None:
                raise RuntimeError(
                    "Geometry must be defined "
                    "before the mesh."
                )

            width = case.geometry.width

            case.mesh = Cartesian1D(
                nx=nx,
                width=width
            )

        else:

            raise ValueError(
                f"Unknown mesh type: "
                f"{mesh_type}"
            )

    # ======================================================
    # MATERIAL
    # ======================================================

    def _read_material(self, case):

        section = self.sections.get("MATERIAL")

        if section is None:
            return

        material = Material(
            name=section["name"],
            D=float(section["D"]),
            Sigma_a=float(section["Sigma_a"]),
            nuSigma_f=float(section["nuSigma_f"]),
        )

        case.materials.add(material)

    # ======================================================
    # PHYSICS
    # ======================================================

    def _read_physics(self, case):

        section = self.sections.get("PHYSICS")

        if section is None:
            return

        physics_type = section.get("type")

        if physics_type == "diffusion":

            groups = int(
                section.get("groups", 1)
            )

            case.physics = Diffusion(
                groups=groups
            )

        else:

            raise ValueError(
                f"Unknown physics type: "
                f"{physics_type}"
            )

    # ======================================================
    # SOLVER
    # ======================================================

    def _read_solver(self, case):

        section = self.sections.get("SOLVER")

        if section is None:
            return

        solver_type = section.get("type")

        if solver_type == "power_iteration":

            tolerance = float(
                section.get(
                    "tolerance",
                    1e-8
                )
            )

            max_iterations = int(
                section.get(
                    "max_iterations",
                    1000
                )
            )

            case.solver = PowerIteration(
                tolerance=tolerance,
                max_iterations=max_iterations,
            )

        else:

            raise ValueError(
                f"Unknown solver type: "
                f"{solver_type}"
            )