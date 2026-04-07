"""Tests for search-cost audit and safe pruning."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pytest

from aria.types import DemoPair, grid_from_list


def _identity_demos() -> tuple[DemoPair, ...]:
    g1 = grid_from_list([[1, 0, 0], [0, 2, 0], [0, 0, 3]])
    g2 = grid_from_list([[4, 0, 0], [0, 5, 0], [0, 0, 6]])
    return (DemoPair(input=g1, output=g1.copy()), DemoPair(input=g2, output=g2.copy()))


class TestSearchCostAudit:
    def test_audit_runs(self):
        from aria.core.guidance_search_cost import audit_search_costs

        pairs = [("t1", _identity_demos()), ("t2", _identity_demos())]
        audit = audit_search_costs(pairs)

        assert audit.n_tasks == 2
        assert audit.total_time_s > 0
        assert len(audit.candidate_costs) > 0
        assert len(audit.candidate_costs) > 0

    def test_audit_to_dict(self):
        from aria.core.guidance_search_cost import audit_search_costs

        pairs = [("t1", _identity_demos())]
        audit = audit_search_costs(pairs)
        d = audit.to_dict()
        serialized = json.dumps(d)
        assert "top_cost_candidate" in json.loads(serialized)

    def test_candidate_costs_have_win_rates(self):
        from aria.core.guidance_search_cost import audit_search_costs

        pairs = [("t1", _identity_demos())]
        audit = audit_search_costs(pairs)

        for c in audit.candidate_costs:
            d = c.to_dict()
            assert "produce_rate" in d
            assert "cost_per_win" in d


class TestPruning:
    def test_pruning_preserves_winner(self):
        from aria.core.guidance_search_cost import evaluate_pruning
        from aria.core.guidance_size_model import train_size_model

        pairs = [("t1", _identity_demos()), ("t2", _identity_demos())]
        model = train_size_model(pairs, epochs=10)

        report, results = evaluate_pruning(pairs, model, confidence_threshold=0.9)

        assert report.n_evaluated > 0
        # With identity tasks (same_dims winner) and never_prune={"same_dims"},
        # winner should always be preserved
        assert report.safety_rate == 1.0

    def test_pruning_report_to_dict(self):
        from aria.core.guidance_search_cost import evaluate_pruning
        from aria.core.guidance_size_model import train_size_model

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=5)
        report, _ = evaluate_pruning(pairs, model)

        d = report.to_dict()
        serialized = json.dumps(d)
        assert "safety_rate" in json.loads(serialized)

    def test_never_prune_respected(self):
        from aria.core.guidance_search_cost import evaluate_pruning
        from aria.core.guidance_size_model import train_size_model

        pairs = [("t1", _identity_demos())]
        model = train_size_model(pairs, epochs=5)
        _, results = evaluate_pruning(
            pairs, model,
            never_prune=frozenset({"same_dims", "scale"}),
        )

        for r in results:
            assert "same_dims" not in r.pruned_families
            assert "scale" not in r.pruned_families


class TestRealARC:
    @pytest.fixture
    def arc_pairs(self):
        try:
            from aria.datasets import get_dataset, list_task_ids, load_arc_task
            ds = get_dataset("v1-train")
            tids = list_task_ids(ds)[:20]
            return [(tid, load_arc_task(ds, tid).train) for tid in tids]
        except Exception:
            pytest.skip("ARC data not available")

    def test_audit_on_real(self, arc_pairs):
        from aria.core.guidance_search_cost import audit_search_costs
        audit = audit_search_costs(arc_pairs)
        assert audit.n_tasks == 20
        assert audit.total_time_s > 0

    def test_pruning_on_real(self, arc_pairs):
        from aria.core.guidance_search_cost import evaluate_pruning
        from aria.core.guidance_size_model import train_size_model

        model = train_size_model(arc_pairs, epochs=30)
        report, _ = evaluate_pruning(arc_pairs, model)
        assert report.n_evaluated > 0


class TestNoTaskIdLogic:
    def test_no_dispatch(self):
        path = Path(__file__).parent.parent / "aria" / "core" / "guidance_search_cost.py"
        content = path.read_text()
        for i, line in enumerate(content.split("\n"), 1):
            s = line.strip()
            if s.startswith("#"):
                continue
            if "task_id" in s and re.search(r'if\s+.*task_id\s*==', s):
                pytest.fail(f"guidance_search_cost.py:{i}: {s}")


class TestNoRegressions:
    def test_output_size_unchanged(self):
        from aria.core.output_size import infer_output_size_spec
        spec = infer_output_size_spec(_identity_demos())
        assert spec is not None
        assert spec.mode == "same_as_input"
