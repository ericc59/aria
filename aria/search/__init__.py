"""Neuro-symbolic search engine for ARC-AGI-2.

Separate from aria/guided/ (the hand-ordered expert pipeline).
Shares perception (perceive.py, correspond.py, output_size.py) but
uses explicit AST programs, canonical execution, and (eventually)
learned search guidance.

Architecture:
  ast.py       — AST IR: Op enum, ASTNode, ASTProgram
  executor.py  — canonical AST executor (no closures)
  search.py    — (future) MCTS / beam over partial ASTs
  propose.py   — (future) learned sketch proposer
  value.py     — (future) learned value model
  macros.py    — (future) abstraction learning / compression
  synthetic.py — (future) synthetic training data generator

Entry point:
  solve.py (at aria/ level) runs guided first, then search on remainder
"""
