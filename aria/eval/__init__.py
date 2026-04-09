"""Compatibility shim for the eval implementation.

Historically the repo ended up with both:
- ``aria/eval.py`` (the actual ARC eval harness)
- ``aria/eval/`` (a package for structural-gates tooling)

Python resolves ``import aria.eval`` to the package, so re-export the
top-level harness implementation explicitly from here until the split is
cleaned up.
"""

from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


_IMPL_PATH = Path(__file__).resolve().parent.parent / "eval.py"
_SPEC = spec_from_file_location("aria._eval_impl", _IMPL_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Unable to load eval implementation from {_IMPL_PATH}")

_MODULE = module_from_spec(_SPEC)
sys.modules.setdefault("aria._eval_impl", _MODULE)
_SPEC.loader.exec_module(_MODULE)


__all__ = [name for name in dir(_MODULE) if not name.startswith("_")]

globals().update({name: getattr(_MODULE, name) for name in __all__})
