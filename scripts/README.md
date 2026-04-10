## Scripts

Current primary entrypoints:

- `solve.py`: canonical single-task solver for the current `aria/search` stack
- `eval_arc2.py`: canonical ARC eval runner for the current `aria/search` stack
- `make_submission.py`: build `submission.json` from eval reports
- `diagnose.py`: summarize eval report failures and gaps
- `test_program.py`: execute a candidate program on a task

Supporting utilities that are still intentionally top-level:

- `run_task.py`: single-task debug runner
- `regression.py`: regression harness
- `prepare_hf_dataset.py`, `train_lora.py`: training and export utilities
- `corpus_report.json`, `extraction_report.json`: retained input artifacts still used as defaults by older utility scripts

Legacy experiment runners and offline-era utilities live under:

- `scripts/legacy/experiments/`
- `scripts/legacy/evals/`
- `scripts/legacy/offline/`
- `scripts/legacy/analysis/`

They are retained for reference, but they are not part of the current
canonical `aria/search` workflow. Older mixed-solver helpers such as
`solve_eval.py` live there as well.
