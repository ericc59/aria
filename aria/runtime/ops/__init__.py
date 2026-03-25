"""Operation registry for the ARIA step machine.

Every operation is registered as (name, type_signature, implementation).
The executor and type checker both consult this registry.
Library abstractions are added as entries at runtime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aria.types import Type


@dataclass(frozen=True)
class OpSignature:
    params: tuple[tuple[str, Type], ...]
    return_type: Type


_REGISTRY: dict[str, tuple[OpSignature, Callable[..., Any]]] = {}
_LIBRARY_OPS: set[str] = set()


def register(
    name: str,
    sig: OpSignature,
    impl: Callable[..., Any],
    *,
    is_library: bool = False,
) -> None:
    if name in _REGISTRY:
        raise ValueError(f"Operation '{name}' already registered")
    _REGISTRY[name] = (sig, impl)
    if is_library:
        _LIBRARY_OPS.add(name)


def get_op(name: str) -> tuple[OpSignature, Callable[..., Any]]:
    if name not in _REGISTRY:
        raise KeyError(f"Unknown operation: '{name}'")
    return _REGISTRY[name]


def has_op(name: str) -> bool:
    return name in _REGISTRY


def all_ops() -> dict[str, OpSignature]:
    return {name: sig for name, (sig, _) in _REGISTRY.items()}


def reset_library_ops() -> None:
    """Remove all non-core ops (for test isolation)."""
    for n in list(_LIBRARY_OPS):
        del _REGISTRY[n]
    _LIBRARY_OPS.clear()
