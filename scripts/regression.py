#!/usr/bin/env python3
"""Quick regression check on known verified solves."""
import numpy as np
from aria.datasets import get_dataset, load_arc_task
from aria.guided.dsl import synthesize_program, _verify

KNOWN_SOLVES = [
    '25ff71a9', '27a77e38', '3618c87e', 'aabf363d',
    '00d62c1b', 'a5313dff', '6d75e8bb', 'd5d6de2d',
    '009d5c81', '05f2a901', '0d3d703e', '00576224', '007bbfb7', '017c7c7b',
    '0c786b71', '0520fde7', '1a6449f1', '1c786137', '34b99a2b', '23b5c85d',
    '1cf80156', '0ca9ddb6', '13713586',
    '1f85a75f', '3af2c5a8', '4258a5f9', '4c4377d9', '0e671a1a', '25d487eb',
    '59341089', '5b6cbef5', '62c24649', '67e8384a', '135a2760',
    '6d0aefbc', '6fa7a44f', '833dafe3', '8be77c9e', '99b1bc43',
    'a2fd1cf0', 'a416b8f3', 'a48eeaf7', 'bc4146bd', 'be94b721',
    'c48954c1', 'c909285e', 'c9e6f938', 'cd3c21df', 'd4a91cb9',
    'd9fac9be', 'dc1df850', 'f0df5ff0',
]

ds_train = get_dataset('v2-train')
ds_eval = get_dataset('v2-eval')
ok = 0
for tid in KNOWN_SOLVES:
    try:
        task = load_arc_task(ds_train, tid)
    except FileNotFoundError:
        task = load_arc_task(ds_eval, tid)
    demos = [(np.array(p.input), np.array(p.output)) for p in task.train]
    test_pairs = [(np.array(p.input), np.array(p.output)) for p in task.test]
    try:
        prog = synthesize_program(demos)
    except Exception as e:
        print(f'FAIL {tid} ERROR: {e}')
        continue
    if prog:
        try:
            tok = _verify(prog, demos)
            tok = tok and all(np.array_equal(prog.execute(ti), to) for ti, to in test_pairs)
        except Exception:
            tok = False
    else:
        tok = False
    status = 'PASS' if tok else 'FAIL'
    print(f'{status} {tid} {prog.description[:80] if prog else "NONE"}')
    if tok:
        ok += 1

print(f'\n{ok}/{len(KNOWN_SOLVES)} passed')
if ok < len(KNOWN_SOLVES):
    exit(1)
