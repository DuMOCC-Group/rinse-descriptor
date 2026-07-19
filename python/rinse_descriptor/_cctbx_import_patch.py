"""Patch cctbx import-time platform issues before importing iotbx/cctbx.

On Linux/Python 3.13, cctbx-base 2025.11 can segfault while initialising
Boost.Python floating-point traps if NumPy has already been imported.  The
Linux guard below disables those traps and preloads Boost.Python before any
``iotbx``/``cctbx`` import.

On Windows, cctbx wheels may contain a bundled ``libtbx_env`` pickle with the
CI build path baked in (for example ``d:\\bld\\cctbx-split_...``).  On machines
where that drive letter is absent or BitLocker-locked, ``os.path.realpath`` can
raise ``OSError`` inside ``libtbx.env_config.unpickle``.  The Windows guard
treats that as a path mismatch so libtbx resets its internal build path to the
current installation location.

Call :func:`patch_cctbx_imports` before any ``iotbx``/``cctbx`` import.
``rinse_descriptor/__init__.py`` does so automatically for normal package use.
"""

from __future__ import annotations

import sys

_PATCHED = False


def patch_cctbx_imports() -> None:
    """Apply cctbx import guards once per Python process."""
    global _PATCHED
    if _PATCHED:
        return

    if sys.platform.startswith("linux"):
        _patch_linux_boost()
    elif sys.platform == "win32":
        _patch_windows_libtbx()

    _PATCHED = True


def _patch_linux_boost() -> None:
    import os

    os.environ["BOOST_ADAPTBX_TRAP_FPE"] = ""
    os.environ["BOOST_ADAPTBX_TRAP_INVALID"] = ""
    os.environ["BOOST_ADAPTBX_TRAP_OVERFLOW"] = ""

    try:
        import boost_adaptbx.boost.python  # noqa: F401
    except Exception:
        pass


def _patch_windows_libtbx() -> None:
    try:
        from collections.abc import Callable
        from typing import Any

        # libtbx.env_config is safe to import: it does not call unpickle()
        # itself.  The problematic call happens later in libtbx.load_env, which
        # is triggered by iotbx.__init__ -> libtbx.version.get_version().
        import libtbx.env_config as env_config

        original_unpickle = env_config.unpickle

        def safe_unpickle(
            build_path: str | None = None,
            env_name: str = "libtbx_env",
            original: Callable[..., Any] = original_unpickle,
        ) -> Any:
            import os.path as op

            original_realpath: Callable[..., str] = op.realpath

            def safe_realpath(path: Any, **kwargs: Any) -> str:
                try:
                    return original_realpath(path, **kwargs)
                except OSError:
                    return str(path)

            op.realpath = safe_realpath  # type: ignore[assignment]
            try:
                return original(build_path=build_path, env_name=env_name)
            finally:
                op.realpath = original_realpath  # type: ignore[assignment]

        env_config.unpickle = safe_unpickle
    except Exception:
        pass
