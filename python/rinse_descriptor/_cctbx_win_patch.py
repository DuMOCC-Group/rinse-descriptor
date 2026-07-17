"""Patch libtbx on Windows to handle OSError from realpath on locked drives.

When cctbx is installed from a conda-forge wheel, the bundled ``libtbx_env``
pickle file contains the CI build path baked in at build time
(e.g. ``d:\\bld\\cctbx-split_...``).  On machines where that drive letter is
absent or BitLocker-locked, ``os.path.realpath`` raises ``OSError`` inside
``libtbx.env_config.unpickle``, preventing any import of ``iotbx`` or
``cctbx``.

This module wraps ``libtbx.env_config.unpickle`` so that an ``OSError`` from
``realpath`` is treated as a path mismatch (causing libtbx to reset its
internal build path to the current installation location, which is the correct
outcome anyway).

**Import this module before any ``iotbx``/``cctbx`` import.**
``rinse_descriptor/__init__.py`` does so automatically.
"""

from __future__ import annotations

import sys

if sys.platform == "win32":
    try:
        # libtbx.env_config is safe to import: it does not call unpickle()
        # itself.  The problematic call happens later in libtbx.load_env, which
        # is triggered by iotbx.__init__ → libtbx.version.get_version().
        import libtbx.env_config as _env_config

        _orig_unpickle = _env_config.unpickle

        def _safe_unpickle(build_path=None, env_name="libtbx_env", _orig=_orig_unpickle):
            import os.path as _op

            _orig_rp = _op.realpath

            def _safe_rp(p, **kw):
                try:
                    return _orig_rp(p, **kw)
                except OSError:
                    return str(p)

            _op.realpath = _safe_rp
            try:
                return _orig(build_path=build_path, env_name=env_name)
            finally:
                _op.realpath = _orig_rp

        _env_config.unpickle = _safe_unpickle
        del _orig_unpickle, _safe_unpickle, _env_config
    except Exception:
        pass
