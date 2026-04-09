## Scripts

Current primary entrypoints:

- `eval_arc2.py`: canonical ARC eval runner for the current guided + `aria/search` stack
- `make_submission.py`: build `submission.json` from eval reports
- `diagnose.py`: summarize eval report failures and gaps
- `test_program.py`: execute a candidate program on a task

Supporting utilities that are still intentionally top-level:

- `run_task.py`: single-task debug runner
- `regression.py`: guided regression harness
- `prepare_hf_dataset.py`, `train_lora.py`: training and export utilities

Legacy experiment runners live under `scripts/legacy/experiments/`.
They are retained for reference, but they are not part of the current
canonical `aria/search` workflow.
