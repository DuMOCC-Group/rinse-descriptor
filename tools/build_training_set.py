"""Build a hyperparameter-optimisation training set from multiple databases.

The training set carries two complementary kinds of supervision so it can
drive the objectives a good descriptor must satisfy:

  * **distinct** structures — chemically diverse structures that the descriptor
    should push *apart*.  Assembled from three databases:

        - 500  *experimental* (not computed) structures from the Materials Project
        - 500  random structures from the Crystallography Open Database (COD)
        - 1000 random structures from the Cambridge Structural Database (CSD)

    Every ``distinct`` structure has a unique reduced chemical formula, so the
    diversity set is chemically well-separated across all three sources.

  * **cluster** groups — sets of structures that *should* embed close together.
    These come from two places:

        - an explicit JSON manifest (``--clusters``); and
        - automatically discovered CSD redeterminations: the largest refcode
          families (by member count) are searched, and members that share a
          Niggli-reduced cell within tolerance (default 0.20 A / 2 deg) are
          treated as redeterminations of one structure.  Only clusters with
          >= 3 members are kept (see ``--n-csd-redet`` for how many families).
          The generous tolerance lets redeterminations across temperatures count
          as the same structure (softer, "graded" positives).

    Cluster members intentionally bypass the formula-uniqueness rule and their
    formulas are reserved so the diversity set never reuses them.

Each structure is stored as a record::

    {"id": "mp:mp-149", "source": "mp", "text": "...", "format": "cif",
     "formula": "SiO2", "group": "<label>",
     "role": "distinct" | "cluster"}

``distinct`` records use their own id as a singleton group label; ``cluster``
records share a group label.  ``format`` is ``"cif"`` for MP/COD
and ``"res"`` for CSD (exported to SHELX RES, the preferred CSD path).  The
records are split group-aware (no structure leaks across the boundary) into:

  * ``--full-output`` (default ``training_set_full.pkl``): the training split,
    read by ``optimise_hyperparams.py`` via ``--full-cache``.
  * ``--test-output`` (default ``training_set_test.pkl``): an unseen held-out
    test split (fraction ``--test-fraction``, default 0.15) for final evaluation.
  * ``--groups-csv`` (default ``groups.csv``): a CSV listing the
    ``(group, refcode, role, source, space_group, cell_volume)`` membership of
    the training split.

Use ``--blacklist`` to drop specific unwanted entries without replacement (a
path to a file of refcodes/ids, or a comma-separated list; cluster groups left
with < 2 members are dropped).

Cluster manifest schema (JSON)::

    {
      "groups": [
        {"label": "silica_polymorphs", "members": [
            {"source": "mp",  "id": "mp-6930"},
            {"source": "cod", "id": "1010938"},
            {"source": "csd", "id": "SIO2AB"}
        ]}
      ]
    }

Usage::

    uv run tools/build_training_set.py --api-key $MP_API_KEY
    uv run tools/build_training_set.py --clusters clusters.json --api-key $MP_API_KEY
    uv run tools/build_training_set.py --dump-csv        # export groups.csv from an existing pickle
    uv run tools/build_training_set.py --trim --blacklist bad.txt  # trim in place
    uv run tools/optimise_hyperparams.py --full-cache training_set_full.pkl

Additional dependencies (install before running):
    uv pip install mp-api pymatgen
    # CSD access requires the CCDC ``ccdc`` Python API and a valid licence.

Only the full and test split pickles are written; the intermediate per-source
diversity results are held in memory and not cached to disk.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import re
import sys
import tempfile
from collections import defaultdict
from functools import reduce
from io import StringIO
from math import gcd
from pathlib import Path
from typing import Any
from zipfile import ZipFile

from libtbx.utils import Sorry  # type: ignore[import-untyped]
from rinse_descriptor import load_cif, load_res

# A single training-set record (see module docstring for the schema).
Record = dict[str, Any]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RECOVERABLE = (
    Sorry,
    ValueError,
    RuntimeError,
    OSError,
    NameError,
    AttributeError,
    KeyError,
)

_DEFAULT_SEED = 42
_DEFAULT_N_MP = 500
_DEFAULT_N_COD = 500
_DEFAULT_N_CSD = 1000
_DEFAULT_N_CSD_REDET = 200  # number of largest refcode families to search
_DEFAULT_TEST_FRACTION = 0.15  # fraction of groups held out as an unseen test set
_DEFAULT_FULL_OUTPUT = Path("training_set_full.pkl")
_DEFAULT_TEST_OUTPUT = Path("training_set_test.pkl")
_DEFAULT_GROUPS_CSV = Path("groups.csv")
_DEFAULT_COD_ZIP = Path(r"C:\Users\Tom\Downloads\cod-cifs-mysql.zip")

_ROLE_DISTINCT = "distinct"
_ROLE_CLUSTER = "cluster"

# Structure text format written per source.  MP/COD ship CIF; CSD is exported to
# SHELX RES (the preferred CSD path, matching compute_csd_hashes.py).
_SOURCE_FORMAT = {"mp": "cif", "cod": "cif", "csd": "res"}

# CSD refcode-family analysis tolerances.  Family members whose Niggli-reduced
# cells agree within these tolerances are treated as redeterminations of the
# same structure (positives that should embed together).  The tolerances are
# deliberately generous so that redeterminations across temperatures count as
# the same structure.
_REDET_LENGTH_TOL = 1  # Angstrom, per reduced-cell edge
_REDET_ANGLE_TOL = 5  # degrees, per reduced-cell angle
_REDET_MIN_CLUSTER = 3  # minimum members for an accepted redetermination cluster

# When fetching, over-sample candidates by this factor to survive structures
# that fail to load or collide with an already-seen formula.
_OVERSAMPLE = 3

_ELEMENT_RE = re.compile(r"^([A-Z][a-z]?)")


# ---------------------------------------------------------------------------
# Canonical formula (shared dedup key across all three databases)
# ---------------------------------------------------------------------------


def _element_from_scattering_type(scattering_type: str) -> str | None:
    """Extract the element symbol from a cctbx scattering-type label.

    Handles labels such as ``"O"``, ``"Ca2+"`` and ``"O2-"``.
    """
    match = _ELEMENT_RE.match(scattering_type.strip())
    return match.group(1) if match else None


def _reduced_formula(xrs: Any) -> str | None:
    """Return a canonical reduced Hill formula for a cctbx xray structure.

    Counts asymmetric-unit scatterers per element, divides by the greatest
    common divisor, then orders C, H first followed by the remaining elements
    alphabetically (Hill convention).  Returns ``None`` if no element can be
    determined.
    """
    counts: dict[str, int] = {}
    for scatterer in xrs.scatterers():
        element = _element_from_scattering_type(scatterer.scattering_type)
        if element is None:
            continue
        counts[element] = counts.get(element, 0) + 1

    if not counts:
        return None

    divisor = reduce(gcd, counts.values())
    if divisor > 1:
        counts = {element: n // divisor for element, n in counts.items()}

    def _hill_key(element: str) -> tuple[int, str]:
        if element == "C":
            return (0, "")
        if element == "H":
            return (1, "")
        return (2, element)

    parts = []
    for element in sorted(counts, key=_hill_key):
        n = counts[element]
        parts.append(element if n == 1 else f"{element}{n}")
    return "".join(parts)


def _formula_from_text(text: str, fmt: str) -> tuple[Any, str] | None:
    """Load a structure string and compute its reduced formula.

    *fmt* selects the parser: ``"cif"`` uses :func:`load_cif`, ``"res"`` uses
    :func:`load_res`.  Returns ``(xrs, formula)`` on success, or ``None`` if the
    structure cannot be loaded or yields no formula.
    """
    loader = load_res if fmt == "res" else load_cif
    try:
        xrs = loader(StringIO(text))
    except _RECOVERABLE:
        return None
    if xrs.scatterers().size() == 0:
        return None
    formula = _reduced_formula(xrs)
    if formula is None:
        return None
    return xrs, formula


def _record(
    entry_id: str,
    source: str,
    text: str,
    fmt: str,
    formula: str,
    group: str,
    role: str,
) -> Record:
    """Build a single training-set record (see module docstring for schema)."""
    return {
        "id": entry_id,
        "source": source,
        "text": text,
        "format": fmt,
        "formula": formula,
        "group": group,
        "role": role,
    }


# ---------------------------------------------------------------------------
# Groups CSV
# ---------------------------------------------------------------------------


def _space_group_and_volume(text: str, fmt: str) -> tuple[str, str]:
    """Return ``(space_group_symbol, cell_volume)`` for a structure.

    Parses the stored structure text; returns empty strings if it cannot be
    loaded.  The volume is in cubic angstroms, formatted to two decimals.
    """
    loader = load_res if fmt == "res" else load_cif
    try:
        xrs = loader(StringIO(text))
        symbol = str(xrs.space_group_info())
        volume = f"{xrs.unit_cell().volume():.2f}"
    except _RECOVERABLE:
        return "", ""
    return symbol, volume


def _write_groups_csv(records: list[Record], path: Path) -> None:
    """Write the group membership of records to CSV.

    Columns: ``group, refcode, role, source, space_group, cell_volume``.  Rows
    are sorted by group then refcode so members of a group are adjacent; space
    group and cell volume are parsed from each stored structure.
    """
    rows = sorted(records, key=lambda r: (str(r["group"]), str(r["id"])))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["group", "refcode", "role", "source", "space_group", "cell_volume"]
        )
        for r in rows:
            sg, vol = _space_group_and_volume(r["text"], r["format"])
            writer.writerow([r["group"], r["id"], r["role"], r["source"], sg, vol])


def _load_blacklist(spec: str | None) -> set[str]:
    """Parse a blacklist from a file path or comma-separated list.

    Accepts either a path to a text file (one id/refcode per line, blank lines
    and ``#`` comments ignored) or a comma-separated string.  Returned tokens
    are lowercased for case-insensitive matching against both the full record
    id (e.g. ``csd:ABCDEF``) and the bare refcode (``ABCDEF``).
    """
    if not spec:
        return set()
    path = Path(spec)
    if path.exists():
        tokens = [
            line.split("#", 1)[0].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
        ]
    else:
        tokens = [t.strip() for t in spec.split(",")]
    return {t.lower() for t in tokens if t}


def _apply_blacklist(
    records: list[Record], blacklist: set[str]
) -> list[Record]:
    """Drop blacklisted entries with no replacement.

    Any record whose id or bare refcode is blacklisted is removed, regardless of
    role.  Cluster groups left with fewer than two members are then dropped
    entirely, since a lone member provides no redetermination pair.
    """
    if not blacklist:
        return records

    def _is_blacklisted(record: Record) -> bool:
        rid = str(record["id"]).lower()
        return rid in blacklist or rid.split(":")[-1] in blacklist

    kept: list[Record] = []
    n_removed = 0
    for record in records:
        if _is_blacklisted(record):
            n_removed += 1
            continue
        kept.append(record)

    # Drop cluster groups that no longer have >= 2 members.
    cluster_counts: dict[str, int] = defaultdict(int)
    for record in kept:
        if record["role"] == _ROLE_CLUSTER:
            cluster_counts[record["group"]] += 1
    thin = {g for g, n in cluster_counts.items() if n < 2}
    n_thinned = 0
    if thin:
        filtered: list[Record] = []
        for record in kept:
            if record["role"] == _ROLE_CLUSTER and record["group"] in thin:
                n_thinned += 1
                continue
            filtered.append(record)
        kept = filtered

    print(
        f"[blacklist] removed {n_removed} blacklisted entr(y/ies); "
        f"dropped {n_thinned} orphaned member(s) from under-populated clusters.",
        file=sys.stderr,
    )
    return kept


# ---------------------------------------------------------------------------
# Materials Project (experimental structures only)
# ---------------------------------------------------------------------------


def _resolve_api_key(cli_key: str | None) -> str:
    key = cli_key or os.environ.get("MP_API_KEY") or os.environ.get("PMG_MAPI_KEY")
    if not key:
        raise ValueError(
            "Materials Project API key missing. "
            "Pass --api-key or set the MP_API_KEY environment variable."
        )
    return key


def _chunks(items: list[Any], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _doc_get(doc: Any, key: str) -> Any:
    if isinstance(doc, dict):
        return doc.get(key)
    return getattr(doc, key, None)


def _dataset_columns(docs: Any, fields: list[str]) -> dict[str, list[Any]]:
    """Extract requested columns from MP API responses without row iteration."""
    if hasattr(docs, "delta_table"):
        table = docs.delta_table
        if hasattr(table, "to_pyarrow_table"):
            return table.to_pyarrow_table(columns=fields).to_pydict()
    if hasattr(docs, "pyarrow_dataset"):
        table = docs.pyarrow_dataset.to_table(columns=fields)
        return table.to_pydict()
    out: dict[str, list[Any]] = {field: [] for field in fields}
    for doc in docs:
        for field in fields:
            out[field].append(_doc_get(doc, field))
    return out


def _collect_mp(
    api_key: str,
    n: int,
    seen: set[str],
    rng: random.Random,
    fetch_batch_size: int = 200,
) -> list[Record]:
    """Collect *n* experimental MP structures with distinct reduced formulas."""
    from mp_api.client import MPRester  # pyright: ignore[reportMissingImports]
    from pymatgen.io.cif import CifWriter  # pyright: ignore[reportMissingImports]

    # The MP server rejects an unbounded summary query (fetching the entire
    # experimental catalog) and caps the page size at 1000, so we bound the
    # request with a valid ``chunk_size`` and ``num_chunks``.  We fetch a pool
    # comfortably larger than the candidate cap so the shuffle below still
    # yields a varied sample.
    id_chunk_size = 1000
    pool_size = max(n * _OVERSAMPLE * 10, 20000)
    num_chunks = max(1, -(-pool_size // id_chunk_size))
    print("[MP] Fetching experimental material IDs...", file=sys.stderr)
    with MPRester(api_key) as mpr:
        docs = mpr.materials.summary.search(
            theoretical=False,  # experimental provenance only (not computed)
            fields=["material_id"],
            chunk_size=id_chunk_size,
            num_chunks=num_chunks,
        )
        columns = _dataset_columns(docs, ["material_id"])
    ids: list[str] = [
        str(mid) for mid in columns.get("material_id", []) if mid is not None
    ]
    print(f"[MP] Found {len(ids)} experimental material IDs", file=sys.stderr)

    rng.shuffle(ids)
    # Cap the candidate pool so we do not fetch the entire database.
    candidates = ids[: max(n * _OVERSAMPLE, n)]

    collected: list[Record] = []
    for batch in _chunks(candidates, fetch_batch_size):
        if len(collected) >= n:
            break
        with MPRester(api_key) as mpr:
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "structure"],
                chunk_size=max(len(batch), 1),
            )
            columns = _dataset_columns(docs, ["material_id", "structure"])
        structures = list(
            zip(
                columns.get("material_id", []),
                columns.get("structure", []),
                strict=False,
            )
        )
        for mid, structure in structures:
            if len(collected) >= n:
                break
            if structure is None or mid is None:
                continue
            try:
                cif_text = str(CifWriter(structure))
            except Exception:
                continue
            result = _formula_from_text(cif_text, "cif")
            if result is None:
                continue
            _, formula = result
            if formula in seen:
                continue
            seen.add(formula)
            entry_id = f"mp:{mid}"
            collected.append(
                _record(
                    entry_id, "mp", cif_text, "cif", formula, entry_id, _ROLE_DISTINCT
                )
            )
        print(
            f"[MP] Collected {len(collected)}/{n} distinct-formula structures",
            file=sys.stderr,
        )

    if len(collected) < n:
        print(
            f"[MP] WARNING: only {len(collected)}/{n} collected "
            "(candidate pool exhausted).",
            file=sys.stderr,
        )
    return collected


# ---------------------------------------------------------------------------
# Crystallography Open Database (from local CIF ZIP archive)
# ---------------------------------------------------------------------------


def _iter_cif_members(zip_file: ZipFile):
    for member in zip_file.infolist():
        if member.is_dir():
            continue
        if member.filename.lower().endswith(".cif"):
            yield member


def _decode_cif_bytes(raw: bytes) -> str:
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _collect_cod(
    zip_path: Path,
    n: int,
    seen: set[str],
    rng: random.Random,
) -> list[Record]:
    """Collect *n* random COD structures with distinct reduced formulas."""
    if not zip_path.exists():
        raise FileNotFoundError(f"COD ZIP archive not found: {zip_path}")

    print(f"[COD] Indexing CIF members in {zip_path}...", file=sys.stderr)
    with ZipFile(zip_path) as zf:
        members = list(_iter_cif_members(zf))
        print(f"[COD] Found {len(members)} CIF files", file=sys.stderr)
        rng.shuffle(members)

        collected: list[Record] = []
        checked = 0
        for member in members:
            if len(collected) >= n:
                break
            checked += 1
            cod_id = Path(member.filename).stem
            try:
                with zf.open(member) as handle:
                    cif_text = _decode_cif_bytes(handle.read())
            except Exception:
                continue
            result = _formula_from_text(cif_text, "cif")
            if result is None:
                continue
            _, formula = result
            if formula in seen:
                continue
            seen.add(formula)
            entry_id = f"cod:{cod_id}"
            collected.append(
                _record(
                    entry_id, "cod", cif_text, "cif", formula, entry_id, _ROLE_DISTINCT
                )
            )
            if len(collected) % 50 == 0:
                print(
                    f"[COD] Collected {len(collected)}/{n} "
                    f"(checked {checked})",
                    file=sys.stderr,
                )

    if len(collected) < n:
        print(
            f"[COD] WARNING: only {len(collected)}/{n} collected "
            "(archive exhausted).",
            file=sys.stderr,
        )
    return collected


# ---------------------------------------------------------------------------
# Cambridge Structural Database (via CCDC ccdc API)
# ---------------------------------------------------------------------------


def _entry_to_res_string(entry: Any) -> str | None:
    """Convert a CSD entry to SHELX RES text using the CCDC writer API."""
    from ccdc.io import CrystalWriter  # pyright: ignore[reportMissingImports]

    fd, tmp_name = tempfile.mkstemp(suffix=".res")
    os.close(fd)
    Path(tmp_name).unlink(missing_ok=True)
    try:
        with CrystalWriter(tmp_name, format="res") as writer:
            writer.write(entry.crystal)
        res_string = Path(tmp_name).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None
    finally:
        Path(tmp_name).unlink(missing_ok=True)
    if not res_string.rstrip().endswith("END"):
        res_string = f"{res_string.rstrip()}\nEND\n"
    return res_string


def _collect_csd(
    n: int,
    seen: set[str],
    rng: random.Random,
) -> list[Record]:
    """Collect *n* random CSD structures with distinct reduced formulas."""
    from ccdc.io import EntryReader  # pyright: ignore[reportMissingImports]

    print("[CSD] Opening CSD database...", file=sys.stderr)
    reader = EntryReader("CSD")
    total = len(reader)
    print(f"[CSD] Database contains {total} entries", file=sys.stderr)

    indices = list(range(total))
    rng.shuffle(indices)

    collected: list[Record] = []
    checked = 0
    for idx in indices:
        if len(collected) >= n:
            break
        checked += 1
        try:
            entry = reader[idx]
        except Exception:
            continue
        res_text = _entry_to_res_string(entry)
        if res_text is None:
            continue
        result = _formula_from_text(res_text, "res")
        if result is None:
            continue
        _, formula = result
        if formula in seen:
            continue
        seen.add(formula)
        entry_id = f"csd:{entry.identifier}"
        collected.append(
            _record(entry_id, "csd", res_text, "res", formula, entry_id, _ROLE_DISTINCT)
        )
        if len(collected) % 50 == 0:
            print(
                f"[CSD] Collected {len(collected)}/{n} (checked {checked})",
                file=sys.stderr,
            )

    if len(collected) < n:
        print(
            f"[CSD] WARNING: only {len(collected)}/{n} collected "
            "(database exhausted).",
            file=sys.stderr,
        )
    return collected


# ---------------------------------------------------------------------------
# CSD refcode families (redeterminations)
# ---------------------------------------------------------------------------
#
# A CSD refcode family (six-letter stem shared by e.g. ABEBUF, ABEBUF01,
# ABEBUF02) collects entries for the same compound.  Within a family we compare
# reduced (Niggli) cells: members with matching cells are redeterminations of the
# same structure (positives that should embed *together*).


def _refcode_family(identifier: str) -> str:
    """Return the six-letter CSD refcode family stem for an identifier.

    CSD refcodes are a six-letter base optionally followed by a two-digit
    redetermination suffix (``ABEBUF`` -> ``ABEBUF``, ``ABEBUF01`` -> ``ABEBUF``);
    family members share the six-letter stem.
    """
    return identifier[:6]


def _reduced_cell_params(xrs: Any) -> tuple[float, ...] | None:
    """Return Niggli-reduced cell parameters (a, b, c, alpha, beta, gamma).

    Reduction gives a canonical basis so cells recorded in different settings are
    compared consistently.  Returns ``None`` if reduction fails.
    """
    try:
        niggli = xrs.unit_cell().niggli_cell()
        return tuple(niggli.parameters())
    except Exception:
        return None


def _reduced_cells_match(
    a: tuple[float, ...],
    b: tuple[float, ...],
    length_tol: float,
    angle_tol: float,
) -> bool:
    """True if two reduced cells agree within edge (A) and angle (deg) tolerances."""
    for i in range(3):
        if abs(a[i] - b[i]) > length_tol:
            return False
    for i in range(3, 6):
        if abs(a[i] - b[i]) > angle_tol:
            return False
    return True


def _cluster_family_by_cell(
    members: list[dict[str, Any]],
    length_tol: float,
    angle_tol: float,
) -> list[list[dict[str, Any]]]:
    """Group family members into connected components of matching reduced cells.

    Each member carries a ``cell`` key.  Two members are joined when their reduced
    cells match within tolerance; the connected components are the clusters.
    """
    n = len(members)
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for i in range(n):
        for j in range(i + 1, n):
            if _reduced_cells_match(
                members[i]["cell"], members[j]["cell"], length_tol, angle_tol
            ):
                union(i, j)

    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for i in range(n):
        buckets[find(i)].append(members[i])
    return list(buckets.values())


def _collect_csd_families(
    n_families: int,
    seen: set[str],
    length_tol: float,
    angle_tol: float,
    min_cluster: int,
) -> list[Record]:
    """Mine the largest CSD refcode families for redeterminations.

    Scans the CSD, groups entries into refcode families, and processes the
    *n_families* families with the most members.  Within each family the members'
    Niggli-reduced cells are clustered with tolerance *length_tol* (A) /
    *angle_tol* (deg):

      * a cell-cluster with at least *min_cluster* members is a set of
        redeterminations of one structure and is emitted as a ``cluster`` record
        group (positives that should embed together).

    Each member's formula is reserved in *seen* so the diversity set never
    reuses it.
    """
    from ccdc.io import EntryReader  # pyright: ignore[reportMissingImports]

    print("[CSD-fam] Opening CSD database...", file=sys.stderr)
    reader = EntryReader("CSD")
    total = len(reader)
    print(
        f"[CSD-fam] Grouping {total} entries into refcode families...",
        file=sys.stderr,
    )

    # Pass 1: group identifiers by refcode family (identifier metadata is cheap).
    families: dict[str, list[str]] = defaultdict(list)
    for idx in range(total):
        try:
            identifier = reader[idx].identifier
        except Exception:
            continue
        families[_refcode_family(identifier)].append(identifier)
        if (idx + 1) % 100000 == 0:
            print(f"[CSD-fam]   scanned {idx + 1}/{total}", file=sys.stderr)

    # Take the families with the most members (largest first).
    multi = sorted(
        (ids for ids in families.values() if len(ids) > 1),
        key=len,
        reverse=True,
    )
    print(
        f"[CSD-fam] {len(multi)} of {len(families)} families have >1 member; "
        f"taking the {min(n_families, len(multi))} largest",
        file=sys.stderr,
    )
    multi = multi[:n_families]

    # Pass 2: for the largest families, load structures and cluster by cell.
    records: list[Record] = []
    n_redet = 0
    for family_ids in multi:
        members: list[dict[str, Any]] = []
        for identifier in family_ids:
            try:
                entry = reader.entry(identifier)
            except Exception:
                continue
            res_text = _entry_to_res_string(entry)
            if res_text is None:
                continue
            result = _formula_from_text(res_text, "res")
            if result is None:
                continue
            xrs, formula = result
            cell = _reduced_cell_params(xrs)
            if cell is None:
                continue
            members.append(
                {"id": identifier, "text": res_text, "formula": formula, "cell": cell}
            )
        if len(members) < 2:
            continue

        cell_clusters = _cluster_family_by_cell(members, length_tol, angle_tol)
        family = _refcode_family(family_ids[0])

        # Redetermination positives: cell-clusters with enough members.
        redet_clusters = [c for c in cell_clusters if len(c) >= min_cluster]
        for k, cluster in enumerate(redet_clusters):
            label = (
                f"csd-redet:{family}"
                if len(redet_clusters) == 1
                else f"csd-redet:{family}-{k}"
            )
            for member in cluster:
                seen.add(member["formula"])
                records.append(
                    _record(
                        f"csd:{member['id']}",
                        "csd",
                        member["text"],
                        "res",
                        member["formula"],
                        label,
                        _ROLE_CLUSTER,
                    )
                )
            n_redet += 1
            print(
                f"[CSD-fam] redet cluster {n_redet}: {label} "
                f"({len(cluster)} members)",
                file=sys.stderr,
            )

    print(
        f"[CSD-fam] Collected {n_redet} redetermination clusters "
        f"({len(records)} member structures)",
        file=sys.stderr,
    )
    return records


# ---------------------------------------------------------------------------
# Cluster groups (structures that should embed *together*)
# ---------------------------------------------------------------------------
#
# Cluster groups are supplied explicitly via a JSON manifest (see the module
# docstring for the schema).  Members are fetched by id from their source, so
# the same single-structure fetch helpers below can be reused by any future
# clustering-aware workflow.  Unlike the diversity sources, cluster members do
# NOT dedup on formula (redeterminations share a formula) — instead their
# formulas are reserved in *seen* so the diversity set never reuses them.


def _fetch_mp_cifs(api_key: str, ids: list[str]) -> dict[str, str]:
    """Fetch CIF text for specific MP material ids, keyed by id."""
    from mp_api.client import MPRester  # pyright: ignore[reportMissingImports]
    from pymatgen.io.cif import CifWriter  # pyright: ignore[reportMissingImports]

    out: dict[str, str] = {}
    with MPRester(api_key) as mpr:
        for batch in _chunks(ids, 200):
            docs = mpr.materials.summary.search(
                material_ids=batch,
                fields=["material_id", "structure"],
                chunk_size=max(len(batch), 1),
            )
            columns = _dataset_columns(docs, ["material_id", "structure"])
            for mid, structure in zip(
                columns.get("material_id", []),
                columns.get("structure", []),
                strict=False,
            ):
                if mid is None or structure is None:
                    continue
                try:
                    out[str(mid)] = str(CifWriter(structure))
                except Exception:
                    continue
    return out


def _fetch_cod_cifs(zip_path: Path, ids: list[str]) -> dict[str, str]:
    """Fetch CIF text for specific COD ids from the ZIP archive, keyed by id."""
    if not zip_path.exists():
        raise FileNotFoundError(f"COD ZIP archive not found: {zip_path}")
    wanted = set(ids)
    out: dict[str, str] = {}
    with ZipFile(zip_path) as zf:
        for member in _iter_cif_members(zf):
            stem = Path(member.filename).stem
            if stem not in wanted:
                continue
            try:
                with zf.open(member) as handle:
                    out[stem] = _decode_cif_bytes(handle.read())
            except Exception:
                continue
            if len(out) == len(wanted):
                break
    return out


def _fetch_csd_texts(refcodes: list[str]) -> dict[str, str]:
    """Fetch SHELX RES text for specific CSD refcodes, keyed by refcode."""
    from ccdc.io import EntryReader  # pyright: ignore[reportMissingImports]

    reader = EntryReader("CSD")
    out: dict[str, str] = {}
    for refcode in refcodes:
        try:
            entry = reader.entry(refcode)
        except Exception:
            continue
        res_text = _entry_to_res_string(entry)
        if res_text is not None:
            out[refcode] = res_text
    return out


_CLUSTER_FETCHERS = {"mp", "cod", "csd"}


def _collect_clusters(
    manifest_path: Path,
    resolve_api_key,
    cod_zip: Path,
    seen: set[str],
) -> list[Record]:
    """Load cluster groups from a JSON manifest into ``cluster`` records.

    *resolve_api_key* is a zero-argument callable returning the MP API key; it
    is only invoked if the manifest references MP members.
    """
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)

    groups = manifest.get("groups", [])
    if not groups:
        print(f"[cluster] No groups found in {manifest_path}", file=sys.stderr)
        return []

    # Gather the ids to fetch per source across all groups.
    ids_by_source: dict[str, set[str]] = {src: set() for src in _CLUSTER_FETCHERS}
    for group in groups:
        for member in group.get("members", []):
            source = member["source"]
            if source not in _CLUSTER_FETCHERS:
                raise ValueError(
                    f"Unknown cluster member source {source!r} "
                    f"(expected one of {sorted(_CLUSTER_FETCHERS)})."
                )
            ids_by_source[source].add(str(member["id"]))

    # Fetch structure text per source (only touching referenced sources).
    text_by_source: dict[str, dict[str, str]] = {src: {} for src in _CLUSTER_FETCHERS}
    if ids_by_source["mp"]:
        print(
            f"[cluster] Fetching {len(ids_by_source['mp'])} MP members...",
            file=sys.stderr,
        )
        text_by_source["mp"] = _fetch_mp_cifs(resolve_api_key(), sorted(ids_by_source["mp"]))
    if ids_by_source["cod"]:
        print(
            f"[cluster] Fetching {len(ids_by_source['cod'])} COD members...",
            file=sys.stderr,
        )
        text_by_source["cod"] = _fetch_cod_cifs(cod_zip, sorted(ids_by_source["cod"]))
    if ids_by_source["csd"]:
        print(
            f"[cluster] Fetching {len(ids_by_source['csd'])} CSD members...",
            file=sys.stderr,
        )
        text_by_source["csd"] = _fetch_csd_texts(sorted(ids_by_source["csd"]))

    records: list[Record] = []
    for group in groups:
        label = str(group["label"])
        group_records: list[Record] = []
        for member in group.get("members", []):
            source = member["source"]
            member_id = str(member["id"])
            fmt = _SOURCE_FORMAT[source]
            text = text_by_source[source].get(member_id)
            if text is None:
                print(
                    f"[cluster] WARNING: {source}:{member_id} in group "
                    f"{label!r} could not be fetched; skipping.",
                    file=sys.stderr,
                )
                continue
            result = _formula_from_text(text, fmt)
            if result is None:
                print(
                    f"[cluster] WARNING: {source}:{member_id} in group "
                    f"{label!r} failed to load; skipping.",
                    file=sys.stderr,
                )
                continue
            _, formula = result
            # Reserve this formula so diversity structures never reuse it.
            seen.add(formula)
            group_records.append(
                _record(
                    f"{source}:{member_id}",
                    source,
                    text,
                    fmt,
                    formula,
                    label,
                    _ROLE_CLUSTER,
                )
            )
        if len(group_records) < 2:
            print(
                f"[cluster] WARNING: group {label!r} has "
                f"{len(group_records)} usable member(s); a cluster needs >= 2.",
                file=sys.stderr,
            )
        records.extend(group_records)

    print(
        f"[cluster] Collected {len(records)} members across {len(groups)} groups",
        file=sys.stderr,
    )
    return records


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------


def _split_key(record: Record) -> str:
    """Return the grouping key that must stay wholly within one split.

    ``distinct`` structures split individually.  Cluster groups split as a unit,
    and all records derived from a single CSD refcode family (its ``csd-redet:``
    groups) split together to avoid leaking a structure across the train/test
    boundary.
    """
    if record["role"] == _ROLE_DISTINCT:
        return f"id:{record['id']}"
    group = record["group"]
    if group.startswith("csd-redet:"):
        family = group.split(":", 1)[1].split("-", 1)[0]
        return f"csdfam:{family}"
    return f"group:{group}"


def _split_train_test(
    records: list[Record],
    test_fraction: float,
    seed: int,
) -> tuple[list[Record], list[Record]]:
    """Partition records into train/test by split key (no group leakage)."""
    if test_fraction <= 0.0:
        return records, []
    keys = sorted({_split_key(r) for r in records})
    rng = random.Random(seed)
    test_keys = {key for key in keys if rng.random() < test_fraction}
    train = [r for r in records if _split_key(r) not in test_keys]
    test = [r for r in records if _split_key(r) in test_keys]
    return train, test


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a mixed MP/COD/CSD training set for hyperparameter optimisation."
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Materials Project API key (prefer env var MP_API_KEY).",
    )
    parser.add_argument(
        "--cod-zip",
        type=Path,
        default=_DEFAULT_COD_ZIP,
        help=f"Path to COD CIF ZIP archive (default: {_DEFAULT_COD_ZIP}).",
    )
    parser.add_argument(
        "--n-mp",
        type=int,
        default=_DEFAULT_N_MP,
        help=f"Number of experimental MP structures (default: {_DEFAULT_N_MP}).",
    )
    parser.add_argument(
        "--n-cod",
        type=int,
        default=_DEFAULT_N_COD,
        help=f"Number of random COD structures (default: {_DEFAULT_N_COD}).",
    )
    parser.add_argument(
        "--n-csd",
        type=int,
        default=_DEFAULT_N_CSD,
        help=f"Number of random CSD structures (default: {_DEFAULT_N_CSD}).",
    )
    parser.add_argument(
        "--n-csd-redet",
        type=int,
        default=_DEFAULT_N_CSD_REDET,
        help=(
            "Number of largest CSD refcode families to search for "
            f"redeterminations (default: {_DEFAULT_N_CSD_REDET})."
        ),
    )
    parser.add_argument(
        "--redet-length-tol",
        type=float,
        default=_REDET_LENGTH_TOL,
        help=(
            "Reduced-cell edge tolerance (A) for grouping CSD redeterminations "
            f"(default: {_REDET_LENGTH_TOL})."
        ),
    )
    parser.add_argument(
        "--redet-angle-tol",
        type=float,
        default=_REDET_ANGLE_TOL,
        help=(
            "Reduced-cell angle tolerance (deg) for grouping CSD redeterminations "
            f"(default: {_REDET_ANGLE_TOL})."
        ),
    )
    parser.add_argument(
        "--clusters",
        type=Path,
        default=None,
        help=(
            "Optional JSON manifest of cluster groups (structures that should "
            "embed together). See the module docstring for the schema."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=_DEFAULT_SEED,
        help=f"Random seed for sampling (default: {_DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--full-output",
        type=Path,
        default=_DEFAULT_FULL_OUTPUT,
        help=(
            "Training-split record pickle (distinct/cluster roles) "
            f"consumed by optimise_hyperparams.py (default: {_DEFAULT_FULL_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--test-output",
        type=Path,
        default=_DEFAULT_TEST_OUTPUT,
        help=(
            "Held-out (unseen) test-split record pickle for final evaluation "
            f"(default: {_DEFAULT_TEST_OUTPUT})."
        ),
    )
    parser.add_argument(
        "--groups-csv",
        type=Path,
        default=_DEFAULT_GROUPS_CSV,
        help=(
            "CSV listing (group, refcode, role, source) for the training split, "
            f"written alongside the pickles (default: {_DEFAULT_GROUPS_CSV})."
        ),
    )
    parser.add_argument(
        "--dump-csv",
        nargs="?",
        const=_DEFAULT_FULL_OUTPUT,
        type=Path,
        default=None,
        metavar="PICKLE",
        help=(
            "Just write the groups CSV (--groups-csv) from an existing record "
            f"pickle (default: {_DEFAULT_FULL_OUTPUT}) and exit, without fetching."
        ),
    )
    parser.add_argument(
        "--trim",
        nargs="?",
        const=_DEFAULT_FULL_OUTPUT,
        type=Path,
        default=None,
        metavar="PICKLE",
        help=(
            "Apply --blacklist to an existing record pickle "
            f"(default: {_DEFAULT_FULL_OUTPUT}), write the trimmed pickle back in "
            "place, refresh --groups-csv, and exit (no fetching)."
        ),
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=_DEFAULT_TEST_FRACTION,
        help=(
            "Fraction of groups held out as the unseen test set "
            f"(default: {_DEFAULT_TEST_FRACTION}; 0 disables the split)."
        ),
    )
    parser.add_argument(
        "--skip-mp",
        action="store_true",
        help="Skip the Materials Project source.",
    )
    parser.add_argument(
        "--skip-cod",
        action="store_true",
        help="Skip the COD source.",
    )
    parser.add_argument(
        "--skip-csd",
        action="store_true",
        help="Skip the CSD source.",
    )
    parser.add_argument(
        "--skip-csd-redet",
        action="store_true",
        help="Skip the CSD refcode-family (redetermination) search.",
    )
    parser.add_argument(
        "--blacklist",
        default=None,
        metavar="PATH_OR_LIST",
        help=(
            "Exclude specific entries without replacement. Accepts a path to a "
            "text file (one refcode/id per line, '#' comments allowed) or a "
            "comma-separated list. Matches the full id (e.g. csd:ABCDEF) or bare "
            "refcode (ABCDEF); cluster groups left with < 2 members are dropped."
        ),
    )
    args = parser.parse_args()

    # ---- Standalone modes on an existing pickle (no fetching) ----
    if args.dump_csv is not None:
        with open(args.dump_csv, "rb") as fh:
            records = pickle.load(fh)
        _write_groups_csv(records, args.groups_csv)
        print(
            f"Wrote {len(records)} rows from {args.dump_csv} to {args.groups_csv}.",
            file=sys.stderr,
        )
        return

    if args.trim is not None:
        with open(args.trim, "rb") as fh:
            records = pickle.load(fh)
        blacklist = _load_blacklist(args.blacklist)
        if not blacklist:
            print(
                "  [trim] no --blacklist given; nothing to trim.",
                file=sys.stderr,
            )
        n_before = len(records)
        records = _apply_blacklist(records, blacklist)
        if len(records) != n_before:
            with open(args.trim, "wb") as fh:
                pickle.dump(records, fh, protocol=pickle.HIGHEST_PROTOCOL)
        if args.groups_csv:
            _write_groups_csv(records, args.groups_csv)
        print(
            f"Trimmed {args.trim}: {n_before} -> {len(records)} records; "
            f"refreshed {args.groups_csv}.",
            file=sys.stderr,
        )
        return

    # Resolve the MP API key lazily: only required if an MP source is used.
    _key_cache: dict[str, str] = {}

    def resolve_api_key() -> str:
        if "key" not in _key_cache:
            _key_cache["key"] = _resolve_api_key(args.api_key)
        return _key_cache["key"]

    seen: set[str] = set()
    records: list[Record] = []

    # ---- Cluster groups first: reserve their formulas from the diversity set ----
    if args.clusters is not None:
        records.extend(
            _collect_clusters(args.clusters, resolve_api_key, args.cod_zip, seen)
        )

    # ---- CSD refcode families: redeterminations (reserve formulas) ----
    if not args.skip_csd_redet:
        records.extend(
            _collect_csd_families(
                args.n_csd_redet,
                seen,
                args.redet_length_tol,
                args.redet_angle_tol,
                _REDET_MIN_CLUSTER,
            )
        )

    # ---- Materials Project (experimental only) ----
    if not args.skip_mp:
        records.extend(
            _collect_mp(
                resolve_api_key(), args.n_mp, seen, random.Random(args.seed)
            )
        )

    # ---- COD ----
    if not args.skip_cod:
        records.extend(
            _collect_cod(
                args.cod_zip, args.n_cod, seen, random.Random(args.seed + 1)
            )
        )

    # ---- CSD ----
    if not args.skip_csd:
        records.extend(
            _collect_csd(args.n_csd, seen, random.Random(args.seed + 2))
        )

    # ---- Optional blacklist (drop unwanted entries, no replacement) ----
    records = _apply_blacklist(
        records, _load_blacklist(args.blacklist)
    )

    # ---- Split into train / unseen test, then write artefacts ----
    train, test = _split_train_test(records, args.test_fraction, args.seed + 7)

    with open(args.full_output, "wb") as fh:
        pickle.dump(train, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if test:
        with open(args.test_output, "wb") as fh:
            pickle.dump(test, fh, protocol=pickle.HIGHEST_PROTOCOL)
    if args.groups_csv:
        _write_groups_csv(train, args.groups_csv)

    def _summarise(rs: list[Record]) -> str:
        n_distinct = sum(1 for r in rs if r["role"] == _ROLE_DISTINCT)
        n_cluster = sum(1 for r in rs if r["role"] == _ROLE_CLUSTER)
        n_cluster_groups = len(
            {r["group"] for r in rs if r["role"] == _ROLE_CLUSTER}
        )
        by_source = {"mp": 0, "cod": 0, "csd": 0}
        for r in rs:
            by_source[r["source"]] = by_source.get(r["source"], 0) + 1
        return (
            f"{len(rs)} records "
            f"(distinct={n_distinct}, cluster={n_cluster} in {n_cluster_groups} groups; "
            f"MP={by_source['mp']}, COD={by_source['cod']}, CSD={by_source['csd']})"
        )

    print(
        f"\nWrote train split to {args.full_output}: {_summarise(train)}; "
        f"{len(seen)} reserved formulas.",
        file=sys.stderr,
    )
    if test:
        print(
            f"Wrote test split to {args.test_output}: {_summarise(test)}.",
            file=sys.stderr,
        )
    if args.groups_csv:
        print(
            f"Wrote group membership of the train split to {args.groups_csv} "
            f"({len(train)} rows).",
            file=sys.stderr,
        )
    print(
        "Run optimisation with:\n"
        f"    uv run tools/optimise_hyperparams.py --full-cache {args.full_output}"
        + (f" --test-cache {args.test_output}" if test else ""),
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
