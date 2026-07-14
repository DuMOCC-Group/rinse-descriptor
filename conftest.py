# Root conftest.py – import cctbx before pytest initialises any captures or
# locale machinery that conflicts with cctbx's bundled libstdc++.
try:
    from cctbx import xray as _xray  # noqa: F401
    from iotbx import cif as _cif  # noqa: F401
except Exception:
    pass
