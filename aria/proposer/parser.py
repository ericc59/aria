"""Parse proposer output text into Program AST.

Accepted statement forms:
    bind name = op(arg1, arg2)
    let name = op(arg1, arg2)
    yield name
    -> name
"""

from __future__ import annotations

import re
from typing import Any

from aria.types import (
    Assert,
    Axis,
    Bind,
    Call,
    Dir,
    Expr,
    Lambda,
    Literal,
    Program,
    Property,
    Ref,
    Shape,
    SizeRank,
    SortDir,
    Step,
    Type,
    ZoneRole,
)


class ParseError(Exception):
    pass


# Maps literal names to (value, type)
_ENUM_LITERALS: dict[str, tuple[Any, Type]] = {}

def _init_enum_literals() -> None:
    # Only register UPPER_CASE names to avoid collisions with variable names.
    # e.g. "frame", "color", "size" are common variables — don't shadow them.
    for d in Dir:
        _ENUM_LITERALS[d.name] = (d, Type.DIR)  # UP, DOWN, LEFT, RIGHT
    for a in Axis:
        _ENUM_LITERALS[a.name] = (a, Type.AXIS)  # HORIZONTAL, VERTICAL, etc.
    for s in Shape:
        _ENUM_LITERALS[s.name] = (s, Type.SHAPE)  # RECT, LINE, DOT, etc.
    for p in Property:
        _ENUM_LITERALS[p.name] = (p, Type.PROPERTY)  # COLOR, SIZE, SHAPE, etc.
    for sd in SortDir:
        _ENUM_LITERALS[sd.name] = (sd, Type.SORT_DIR)  # ASC, DESC
    _ENUM_LITERALS["ascending"] = (SortDir.ASC, Type.SORT_DIR)
    _ENUM_LITERALS["descending"] = (SortDir.DESC, Type.SORT_DIR)
    for sr in SizeRank:
        _ENUM_LITERALS[sr.name] = (sr, Type.SIZE_RANK)  # LARGEST, SMALLEST
    for zr in ZoneRole:
        _ENUM_LITERALS[zr.name] = (zr, Type.ZONE_ROLE)  # RULE, DATA, FRAME, BORDER
    _ENUM_LITERALS["true"] = (True, Type.BOOL)
    _ENUM_LITERALS["false"] = (False, Type.BOOL)
    _ENUM_LITERALS["True"] = (True, Type.BOOL)
    _ENUM_LITERALS["False"] = (False, Type.BOOL)

_init_enum_literals()


# Type name mapping
_TYPE_NAMES: dict[str, Type] = {t.name.lower(): t for t in Type}
_TYPE_NAMES.update({t.name: t for t in Type})
# Common aliases
_TYPE_NAMES["ObjectSet"] = Type.OBJECT_SET
_TYPE_NAMES["ObjectList"] = Type.OBJECT_LIST
_TYPE_NAMES["ColorMap"] = Type.COLOR_MAP
_TYPE_NAMES["IntList"] = Type.INT_LIST
_TYPE_NAMES["ZoneList"] = Type.ZONE_LIST
_TYPE_NAMES["TaskCtx"] = Type.TASK_CTX


def _parse_type(s: str) -> Type:
    s = s.strip()
    if s in _TYPE_NAMES:
        return _TYPE_NAMES[s]
    raise ParseError(f"Unknown type: '{s}'")


def _tokenize_expr(s: str) -> list[str]:
    """Tokenize an expression string into tokens."""
    tokens: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]
        if c in " \t":
            i += 1
        elif (
            c == "-"
            and i + 1 < len(s)
            and s[i + 1].isdigit()
            and (i == 0 or s[i - 1] in " \t(,{|:+-*/")
        ):
            j = i + 1
            while j < len(s) and s[j].isdigit():
                j += 1
            tokens.append(s[i:j])
            i = j
        elif c in "(),+*-/{}:|":
            tokens.append(c)
            i += 1
        else:
            j = i
            while j < len(s) and s[j] not in " \t(),+*-/{}:|":
                j += 1
            tokens.append(s[i:j])
            i = j
    return tokens


def _parse_dict_literal(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Parse {key: val, key: val, ...} as a ColorMap literal."""
    pos += 1  # skip {
    mapping: dict[int, int] = {}
    while pos < len(tokens) and tokens[pos] != "}":
        if tokens[pos] == ",":
            pos += 1
            continue
        # key
        key_tok = tokens[pos]
        pos += 1
        # :
        if pos < len(tokens) and tokens[pos] == ":":
            pos += 1
        # value
        val_tok = tokens[pos]
        pos += 1
        try:
            mapping[int(key_tok)] = int(val_tok)
        except ValueError:
            pass  # skip non-int entries
    if pos < len(tokens) and tokens[pos] == "}":
        pos += 1
    return Literal(value=mapping, typ=Type.COLOR_MAP), pos


def _parse_expr_with_infix(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Parse an expression including infix arithmetic operators."""
    expr, pos = _parse_expr_tokens(tokens, pos)

    while pos < len(tokens) and tokens[pos] in ("+", "-", "*"):
        op = tokens[pos]
        op_name = {"+": "add", "-": "sub", "*": "mul"}[op]
        pos += 1
        right, pos = _parse_expr_tokens(tokens, pos)
        expr = Call(op=op_name, args=(expr, right))

    return expr, pos


def _parse_lambda(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Parse |param: TYPE, ...| body into nested Lambda nodes."""
    pos += 1  # skip opening |
    params: list[tuple[str, Type]] = []

    while pos < len(tokens):
        if tokens[pos] == "|":
            pos += 1
            break

        name = tokens[pos]
        pos += 1
        if pos >= len(tokens) or tokens[pos] != ":":
            raise ParseError("Malformed lambda parameter list")
        pos += 1
        if pos >= len(tokens):
            raise ParseError("Missing lambda parameter type")
        param_type = _parse_type(tokens[pos])
        pos += 1
        params.append((name, param_type))

        if pos < len(tokens) and tokens[pos] == ",":
            pos += 1
            continue
        if pos < len(tokens) and tokens[pos] == "|":
            pos += 1
            break
        raise ParseError("Malformed lambda parameter list")

    if not params:
        raise ParseError("Lambda must declare at least one parameter")

    body, pos = _parse_expr_with_infix(tokens, pos)
    expr: Expr = body
    for name, param_type in reversed(params):
        expr = Lambda(param=name, param_type=param_type, body=expr)
    return expr, pos


def _parse_expr_tokens(tokens: list[str], pos: int) -> tuple[Expr, int]:
    """Parse an expression from tokens starting at pos. Returns (expr, new_pos)."""
    if pos >= len(tokens):
        raise ParseError("Unexpected end of expression")

    tok = tokens[pos]

    if tok == "|":
        return _parse_lambda(tokens, pos)

    # Integer literal
    if tok.isdigit() or (tok.startswith("-") and len(tok) > 1 and tok[1:].isdigit()):
        return Literal(value=int(tok), typ=Type.INT), pos + 1

    # Tuple: (expr, expr, ...)
    if tok == "(":
        pos += 1  # skip (
        exprs: list[Expr] = []
        while pos < len(tokens) and tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            elem, pos = _parse_expr_tokens(tokens, pos)
            exprs.append(elem)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        # If all elements are int literals, return a static tuple
        if all(isinstance(e, Literal) and isinstance(e.value, int) for e in exprs):
            return Literal(value=tuple(e.value for e in exprs), typ=Type.PAIR), pos
        # Otherwise emit make_tuple(...) so elements are evaluated at runtime
        return Call(op="make_tuple", args=tuple(exprs)), pos

    # Dict literal: {key: val, ...}
    if tok == "{":
        return _parse_dict_literal(tokens, pos)

    # Enum literal (UPPER_CASE only)
    if tok in _ENUM_LITERALS:
        val, typ = _ENUM_LITERALS[tok]
        return Literal(value=val, typ=typ), pos + 1

    # Name or function call
    if pos + 1 < len(tokens) and tokens[pos + 1] == "(":
        # Function call: name(arg1, arg2, ...)
        op_name = tok
        pos += 2  # skip name and (
        args: list[Expr] = []
        while pos < len(tokens) and tokens[pos] != ")":
            if tokens[pos] == ",":
                pos += 1
                continue
            arg, pos = _parse_expr_tokens(tokens, pos)
            args.append(arg)
        if pos < len(tokens) and tokens[pos] == ")":
            pos += 1
        return Call(op=op_name, args=tuple(args)), pos

    # Name reference
    return Ref(name=tok), pos + 1


def parse_expr(s: str) -> Expr:
    """Parse an expression string into an Expr AST node."""
    tokens = _tokenize_expr(s.strip())
    if not tokens:
        raise ParseError(f"Empty expression: '{s}'")
    expr, _ = _parse_expr_with_infix(tokens, 0)
    return expr


def parse_program(text: str) -> Program:
    """Parse a program text into a Program AST.

    Format:
        bind name : Type = expr
        bind name = expr           (type inferred later)
        let name : Type = expr
        let name = expr            (type inferred later)
        assert pred
        yield name
        -> name
    """
    lines = [line.strip() for line in text.strip().splitlines()]
    lines = [line for line in lines if line and not line.startswith("--")]

    steps: list[Step] = []
    output: str | None = None

    for line in lines:
        if line.startswith("yield "):
            output = line[6:].strip()
        elif line.startswith("-> "):
            output = line[3:].strip()
        elif line.startswith(("bind ", "let ")):
            rest = line[5:] if line.startswith("bind ") else line[4:]
            # Try: name : Type = expr
            match = re.match(r"(\w+)\s*:\s*(\w+)\s*=\s*(.+)", rest)
            if match:
                name = match.group(1)
                typ = _parse_type(match.group(2))
                expr = parse_expr(match.group(3))
                steps.append(Bind(name=name, typ=typ, expr=expr, declared=True))
                continue

            # Try: name = expr (type defaults to GRID)
            match = re.match(r"(\w+)\s*=\s*(.+)", rest)
            if match:
                name = match.group(1)
                expr = parse_expr(match.group(2))
                # Placeholder type; type_check will infer the real binding type.
                steps.append(Bind(name=name, typ=Type.GRID, expr=expr, declared=False))
                continue

            raise ParseError(f"Malformed bind: '{line}'")

        elif line.startswith("assert "):
            pred = parse_expr(line[7:])
            steps.append(Assert(pred=pred))
        else:
            raise ParseError(f"Unknown statement: '{line}'")

    if output is None:
        raise ParseError("No yield statement found")

    return Program(steps=tuple(steps), output=output)
