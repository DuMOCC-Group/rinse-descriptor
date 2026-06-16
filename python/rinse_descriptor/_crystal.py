"""Internal Crystal dataclass – ASE/Gemmi-independent representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    pass


@dataclass
class Crystal:
    """Minimal crystallographic representation passed to descriptor kernels.

    Attributes
    ----------
    cell:
        (3, 3) float64 array, rows are lattice vectors **a**, **b**, **c** in Å.
    positions:
        (N, 3) float64 array of Cartesian atomic positions in Å.
    species:
        (N,) array of atomic numbers (int32).
    occupancies:
        (N,) float64 array of site occupancies.
    u_iso:
        (N,) float64 array of isotropic displacement values (U_iso, Å²).
    u_aniso:
        (N, 3, 3) float64 array of anisotropic displacement tensors (U_ij, Å²)
        in CIF convention when available.
    pbc:
        (3,) bool array; True along each axis that has periodic boundary conditions.
    """

    cell: NDArray[np.float64]
    positions: NDArray[np.float64]
    species: NDArray[np.int32]
    occupancies: NDArray[np.float64] | None = None
    u_iso: NDArray[np.float64] | None = None
    u_aniso: NDArray[np.float64] | None = None
    pbc: NDArray[np.bool_] = field(
        default_factory=lambda: np.array([True, True, True], dtype=np.bool_)
    )
    gemmi_small_structure: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.cell = np.asarray(self.cell, dtype=np.float64)
        self.positions = np.asarray(self.positions, dtype=np.float64)
        self.species = np.asarray(self.species, dtype=np.int32)
        self.pbc = np.asarray(self.pbc, dtype=np.bool_)

        n_atoms = int(self.positions.shape[0])

        if self.occupancies is None:
            self.occupancies = np.ones(n_atoms, dtype=np.float64)
        else:
            self.occupancies = np.asarray(self.occupancies, dtype=np.float64)

        if self.u_iso is None:
            self.u_iso = np.zeros(n_atoms, dtype=np.float64)
        else:
            self.u_iso = np.asarray(self.u_iso, dtype=np.float64)

        if self.u_aniso is None:
            self.u_aniso = np.zeros((n_atoms, 3, 3), dtype=np.float64)
        else:
            self.u_aniso = np.asarray(self.u_aniso, dtype=np.float64)

        if self.cell.shape != (3, 3):
            raise ValueError(f"cell must be (3, 3), got {self.cell.shape}")
        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(f"positions must be (N, 3), got {self.positions.shape}")
        if self.species.shape != (self.positions.shape[0],):
            raise ValueError(
                "species length "
                f"{self.species.shape} must match positions {self.positions.shape[0]}"
            )
        if self.occupancies.shape != (n_atoms,):
            raise ValueError(
                f"occupancies must have shape ({n_atoms},), got {self.occupancies.shape}"
            )
        if self.u_iso.shape != (n_atoms,):
            raise ValueError(f"u_iso must have shape ({n_atoms},), got {self.u_iso.shape}")
        if self.u_aniso.shape != (n_atoms, 3, 3):
            raise ValueError(f"u_aniso must have shape ({n_atoms}, 3, 3), got {self.u_aniso.shape}")
        if self.pbc.shape != (3,):
            raise ValueError(f"pbc must be (3,), got {self.pbc.shape}")

    @property
    def n_atoms(self) -> int:
        """Number of atoms."""
        return int(self.positions.shape[0])

    @property
    def volume(self) -> float:
        """Unit-cell volume in Å³."""
        return float(abs(np.linalg.det(self.cell)))

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["gemmi_small_structure"] = None
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_ase(cls, atoms: object) -> Crystal:
        """Build a Crystal from an :class:`ase.Atoms` object."""
        try:
            import ase  # noqa: F401
            from ase import Atoms as AseAtoms
        except ImportError as exc:
            raise ImportError("ase is required to use Crystal.from_ase()") from exc

        if not isinstance(atoms, AseAtoms):
            raise TypeError(f"Expected ase.Atoms, got {type(atoms)}")

        return cls(
            cell=np.array(atoms.cell[:], dtype=np.float64),
            positions=np.array(atoms.get_positions(), dtype=np.float64),  # type: ignore[no-untyped-call]
            species=np.array(atoms.get_atomic_numbers(), dtype=np.int32),  # type: ignore[no-untyped-call]
            pbc=np.array(atoms.get_pbc(), dtype=np.bool_),  # type: ignore[no-untyped-call]
        )

    @classmethod
    def from_cif(cls, path: str) -> Crystal:
        """Load a structure from a CIF file.

        Uses gemmi when available; falls back to a pure-Python CIF parser
        (e.g., on WebAssembly / Emscripten platforms where gemmi has no wheel).

        The asymmetric unit is expanded to the full unit cell using the space
        group symmetry operations stored in (or inferred from) the CIF.
        """
        try:
            from gemmi import Op, cif, find_spacegroup_by_name, make_small_structure_from_block
        except ImportError:
            from ._pure_python import crystal_from_cif_pure
            return crystal_from_cif_pure(path)

        doc = cif.read_file(str(path))
        # Use sole_block() when possible; fall back to first block.
        try:
            block = doc.sole_block()
        except Exception:
            block = doc[0]

        structure = make_small_structure_from_block(block)

        if len(structure.sites) == 0:
            raise ValueError(f"CIF file {path!r} does not contain any atomic sites")

        # --- Cell matrix: rows are lattice vectors a, b, c in Å ---
        # structure.cell.orth.mat is the orthogonalisation matrix M such that
        #   cart = M @ frac  (columns of M are lattice vectors).
        # Crystal.cell convention: rows are lattice vectors, i.e. cell = M.T.
        orth = np.array(structure.cell.orth.mat.tolist(), dtype=np.float64)  # (3, 3)
        cell_matrix = orth.T  # rows = a, b, c

        # --- Space-group operations ---
        sg_hm = str(structure.spacegroup_hm or "P 1").strip()
        sg_obj = find_spacegroup_by_name(sg_hm)
        sg_ops = list(sg_obj.operations()) if sg_obj is not None else [Op("x,y,z")]

        # --- Expand asymmetric unit to full unit cell ---
        TOL = 1e-4
        expanded_frac: list[list[float]] = []
        expanded_Z: list[int] = []
        expanded_occ: list[float] = []
        expanded_u_iso: list[float] = []
        expanded_u_aniso: list[list[list[float]]] = []

        for site in structure.sites:
            xyz0 = [site.fract.x, site.fract.y, site.fract.z]
            Z = int(site.element.atomic_number)
            occ = float(getattr(site, "occ", 1.0))
            u_iso = float(getattr(site, "u_iso", 0.0))
            u_aniso = _extract_site_u_aniso(site)
            seen: list[tuple[float, float, float]] = []
            for op in sg_ops:
                xyz = op.apply_to_xyz(xyz0)
                xyz = [v % 1.0 for v in xyz]  # wrap into [0, 1)
                # Skip duplicates within tolerance
                if any(
                    abs(xyz[0] - s[0]) < TOL
                    and abs(xyz[1] - s[1]) < TOL
                    and abs(xyz[2] - s[2]) < TOL
                    for s in seen
                ):
                    continue
                seen.append((xyz[0], xyz[1], xyz[2]))
                expanded_frac.append(xyz)
                expanded_Z.append(Z)
                expanded_occ.append(occ)
                expanded_u_iso.append(u_iso)
                expanded_u_aniso.append(u_aniso)

        if not expanded_frac:
            raise ValueError(f"CIF file {path!r} did not yield any atoms after symmetry expansion")

        # --- Fractional → Cartesian ---
        frac_arr = np.array(expanded_frac, dtype=np.float64)  # (N, 3)
        positions = frac_arr @ cell_matrix  # (N, 3) Å

        return cls(
            cell=cell_matrix,
            positions=positions,
            species=np.array(expanded_Z, dtype=np.int32),
            occupancies=np.array(expanded_occ, dtype=np.float64),
            u_iso=np.array(expanded_u_iso, dtype=np.float64),
            u_aniso=np.array(expanded_u_aniso, dtype=np.float64),
            pbc=np.array([True, True, True], dtype=np.bool_),
            gemmi_small_structure=structure,
        )


def _extract_site_u_aniso(site: object) -> list[list[float]]:
    """Extraction of anisotropic U_ij from a gemmi site.

    Returns a 3x3 zero matrix when anisotropic terms are absent.
    """
    aniso = getattr(site, "aniso", None)
    if aniso is None:
        return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]

    required = ("u11", "u22", "u33", "u12", "u13", "u23")
    if all(hasattr(aniso, attr) for attr in required):
        u11 = float(getattr(aniso, "u11"))
        u22 = float(getattr(aniso, "u22"))
        u33 = float(getattr(aniso, "u33"))
        u12 = float(getattr(aniso, "u12"))
        u13 = float(getattr(aniso, "u13"))
        u23 = float(getattr(aniso, "u23"))
        return [[u11, u12, u13], [u12, u22, u23], [u13, u23, u33]]

    mat = getattr(aniso, "mat", None)
    if mat is not None and hasattr(mat, "tolist"):
        vals = mat.tolist()
        if isinstance(vals, list) and len(vals) == 3:
            return [[float(v) for v in row] for row in vals]

    return [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
