"""Editable-workspace startup tweaks for native dependencies."""

from __future__ import annotations

import os
import sys

if sys.platform.startswith("linux"):
    # cctbx-base 2025.11 can segfault while initialising Boost.Python floating
    # point traps if NumPy has already been imported by the host process.
    os.environ["BOOST_ADAPTBX_TRAP_FPE"] = ""
    os.environ["BOOST_ADAPTBX_TRAP_INVALID"] = ""
    os.environ["BOOST_ADAPTBX_TRAP_OVERFLOW"] = ""

    try:
        import boost_adaptbx.boost.python  # noqa: F401
    except Exception:
        pass
