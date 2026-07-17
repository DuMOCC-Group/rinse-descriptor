# Root conftest.py – import cctbx before pytest initialises any captures or
# locale machinery that conflicts with cctbx's bundled libstdc++.
#
# Import rinse_descriptor first so that _cctbx_win_patch runs before iotbx is
# touched (required on Windows when the conda-forge build drive is absent).
try:
    import rinse_descriptor as _  # noqa: F401
except Exception:
    pass
try:
    from cctbx import xray as _xray  # noqa: F401
    from iotbx import cif as _cif  # noqa: F401
except Exception:
    pass
