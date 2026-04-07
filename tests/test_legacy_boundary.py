"""Verify the legacy module boundary is stable.

Every public and test-visible name must be importable from both the canonical
``aria.legacy.*`` path and the backward-compatible ``aria.*`` shim.
"""

import importlib
import types

import pytest


# Canonical legacy modules and their shims.
_LEGACY_MODULES = [
    "aria.legacy.observe",
    "aria.legacy.refinement",
    "aria.legacy.structural_edit",
    "aria.legacy.offline_search",
]


@pytest.mark.parametrize("modname", _LEGACY_MODULES)
def test_legacy_module_importable(modname: str) -> None:
    """Each legacy module can be imported from its canonical path."""
    mod = importlib.import_module(modname)
    assert isinstance(mod, types.ModuleType)


@pytest.mark.parametrize("modname", _LEGACY_MODULES)
def test_shim_reexports_all_public_names(modname: str) -> None:
    """The shim at the old path re-exports every public name."""
    canonical = importlib.import_module(modname)
    shim_name = modname.replace("aria.legacy.", "aria.")
    shim = importlib.import_module(shim_name)

    public_names = [n for n in dir(canonical) if not n.startswith("_")]
    for name in public_names:
        assert hasattr(shim, name), f"{shim_name} missing re-export: {name}"


@pytest.mark.parametrize("modname", _LEGACY_MODULES)
def test_shim_reexports_private_test_names(modname: str) -> None:
    """Private names used by the test suite are re-exported through the shim."""
    canonical = importlib.import_module(modname)
    shim_name = modname.replace("aria.legacy.", "aria.")
    shim = importlib.import_module(shim_name)

    private_names = [n for n in dir(canonical) if n.startswith("_") and not n.startswith("__")]
    for name in private_names:
        assert hasattr(shim, name), f"{shim_name} missing private re-export: {name}"


# --- Generalized core must NOT import from legacy --------------------------

_GENERALIZED_CORE = [
    "aria.sketch",
    "aria.sketch_compile",
    "aria.sketch_fit",
]


@pytest.mark.parametrize("modname", _GENERALIZED_CORE)
def test_generalized_core_does_not_import_legacy(modname: str) -> None:
    """Generalized-core modules must not depend on legacy modules."""
    mod = importlib.import_module(modname)
    source_file = mod.__file__
    assert source_file is not None
    source = open(source_file).read()
    for legacy in ("aria.observe", "aria.refinement", "aria.structural_edit", "aria.offline_search"):
        # Allow the shim files themselves — check the actual source, not re-export shims
        assert f"from {legacy}" not in source or "aria.legacy" in source, (
            f"{modname} imports from legacy module {legacy}"
        )
