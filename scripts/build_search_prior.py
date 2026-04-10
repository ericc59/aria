#!/usr/bin/env python3
"""Build the persisted proposal prior for the canonical `aria/search` stack."""

from __future__ import annotations

import argparse
from pathlib import Path

from aria.search.proposal_memory import SearchProposalPrior, default_prior_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a persisted search proposal prior from eval reports",
    )
    parser.add_argument(
        "--report",
        action="append",
        default=[],
        help="Optional eval report path(s). Defaults to results/eval_*.json",
    )
    parser.add_argument(
        "--output",
        default=str(default_prior_path()),
        help="Output JSON path for the persisted search prior",
    )
    args = parser.parse_args()

    report_paths = [Path(p) for p in args.report]
    if report_paths:
        prior = SearchProposalPrior.from_eval_reports(report_paths)
    else:
        from aria.search.proposal_memory import build_default_search_prior

        prior = build_default_search_prior()
        print(f"Wrote {default_prior_path()}")
        print(
            f"Families: {len(prior.global_counts)} | "
            f"Signature buckets: {len(prior.by_signature)}"
        )
        return

    out = Path(args.output)
    prior.save_json(out)
    print(f"Wrote {out}")
    print(
        f"Families: {len(prior.global_counts)} | "
        f"Signature buckets: {len(prior.by_signature)}"
    )


if __name__ == "__main__":
    main()
