"""Tests for budget allocation policy."""

from __future__ import annotations

from aria.core.budget import BudgetPolicy, LaneBudget, allocate_budget, format_budget
from aria.types import DemoPair, grid_from_list


def _simple():
    return (DemoPair(input=grid_from_list([[1, 2]]), output=grid_from_list([[2, 1]])),)


def test_allocate_returns_policy():
    policy = allocate_budget(_simple())
    assert isinstance(policy, BudgetPolicy)
    assert policy.total_compiles > 0


def test_policy_has_lane_budgets():
    policy = allocate_budget(_simple())
    assert "periodic_repair" in policy.lane_budgets or "relocation" in policy.lane_budgets


def test_anti_evidence_gets_zero():
    """Lanes with anti-evidence should get 0 budget."""
    policy = allocate_budget(_simple())
    for name, lb in policy.lane_budgets.items():
        if "skipped" in lb.rationale:
            assert lb.max_combos == 0


def test_budget_cap_respected():
    policy = allocate_budget(_simple(), total_compile_cap=30)
    assert policy.total_compiles <= 30


def test_deterministic():
    p1 = allocate_budget(_simple())
    p2 = allocate_budget(_simple())
    assert p1.total_compiles == p2.total_compiles


def test_format_budget():
    policy = allocate_budget(_simple())
    text = format_budget(policy)
    assert "Budget Policy" in text


def test_no_task_id():
    import inspect
    src = inspect.getsource(allocate_budget)
    assert "1b59e163" not in src
