"""DEPRECATED — Hybrid neural-symbolic solver.

This module is superseded by the canonical graph pipeline
(aria.core.graph + aria.core.protocol). The stepper-op ranking
approach was an experiment that did not produce results on eval
tasks. Preserved for reference only.

The future learned path is the per-task recurrent graph editor
(aria.core.editor_env), not stepper-op ranking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from aria.core.experimental.neural import OpPredictor, TrainingExample, TaskFeatures, extract_features
from aria.core.stepper import (
    StepperResult,
    StepCandidate,
    ConstructionStep,
    _pixel_diff,
    _generate_candidates,
    _apply_candidate,
    _build_program,
    _identity_program,
    _GRID_PLACEHOLDER,
    _BeamEntry,
)
from aria.core.world import WorldModel, build_world_model
from aria.types import DemoPair, Grid, Program
from aria.verify.verifier import verify as aria_verify


@dataclass(frozen=True)
class HybridResult:
    """Outcome of hybrid solving on a set of tasks."""
    total_tasks: int
    solved_tasks: int
    solved_task_ids: tuple[str, ...]
    programs: dict[str, Program]          # task_id -> winning program
    training_examples: int                # examples in predictor
    predictor_trained: bool


class HybridSolver:
    """Neural-symbolic solver with learned candidate prioritization."""

    def __init__(self, beam_width: int = 5, max_steps: int = 6) -> None:
        self.predictor = OpPredictor()
        self.beam_width = beam_width
        self.max_steps = max_steps
        self._solved: dict[str, Program] = {}

    def bootstrap(
        self,
        tasks: Sequence[tuple[str, tuple[DemoPair, ...]]],
    ) -> int:
        """Phase 1: solve with fitters + unguided stepper, build training data."""
        from aria.core.stepper import step_solve, beam_solve

        solved_count = 0
        for task_id, demos in tasks:
            if not all(d.input.shape == d.output.shape for d in demos):
                # Try fitters for dims-change tasks
                prog = self._try_fitters(task_id, demos)
                if prog is not None:
                    self._record_solve(task_id, demos, prog, [])
                    solved_count += 1
                continue

            try:
                world = build_world_model(demos, task_id=task_id)
            except Exception:
                world = None

            # Try greedy stepper first (fast)
            result = step_solve(demos, max_steps=self.max_steps, world=world)
            if result.solved and result.program is not None:
                vr = aria_verify(result.program, demos)
                if vr.passed:
                    self._record_solve(task_id, demos, result.program,
                                       list(result.steps_description))
                    solved_count += 1
                    continue

            # Try beam (slower, deeper)
            result = beam_solve(demos, max_steps=self.max_steps,
                                beam_width=self.beam_width, world=world)
            if result.solved and result.program is not None:
                vr = aria_verify(result.program, demos)
                if vr.passed:
                    self._record_solve(task_id, demos, result.program,
                                       list(result.steps_description))
                    solved_count += 1
                    continue

            # Try fitters as fallback
            prog = self._try_fitters(task_id, demos)
            if prog is not None:
                self._record_solve(task_id, demos, prog, [])
                solved_count += 1

        return solved_count

    def train(self) -> None:
        """Phase 2: train the op predictor from solved tasks."""
        self.predictor.train()

    def solve(
        self,
        tasks: Sequence[tuple[str, tuple[DemoPair, ...]]],
    ) -> HybridResult:
        """Phase 3: solve with learned candidate guidance."""
        new_solved: list[str] = []

        for task_id, demos in tasks:
            if task_id in self._solved:
                continue
            if not all(d.input.shape == d.output.shape for d in demos):
                prog = self._try_fitters(task_id, demos)
                if prog is not None:
                    self._record_solve(task_id, demos, prog, [])
                    new_solved.append(task_id)
                continue

            try:
                world = build_world_model(demos, task_id=task_id)
                features = extract_features(world)
            except Exception:
                world = None
                features = None

            result = self._guided_beam_solve(demos, world, features)

            if result.solved and result.program is not None:
                vr = aria_verify(result.program, demos)
                if vr.passed:
                    self._record_solve(task_id, demos, result.program,
                                       list(result.steps_description))
                    new_solved.append(task_id)

        return HybridResult(
            total_tasks=len(tasks),
            solved_tasks=len(self._solved),
            solved_task_ids=tuple(sorted(self._solved.keys())),
            programs=dict(self._solved),
            training_examples=self.predictor.n_examples,
            predictor_trained=self.predictor._trained,
        )

    def _guided_beam_solve(
        self,
        demos: tuple[DemoPair, ...],
        world: WorldModel | None,
        features: TaskFeatures | None,
    ) -> StepperResult:
        """Beam search with learned candidate ordering."""
        from aria.core.stepper import ConstructionResult

        targets = [d.output for d in demos]
        initial_states = [d.input.copy() for d in demos]
        initial_diff = sum(_pixel_diff(s, t) for s, t in zip(initial_states, targets))

        if initial_diff == 0:
            return StepperResult(solved=True, program=_identity_program())

        beam = [_BeamEntry(states=initial_states, steps=[], total_diff=initial_diff)]

        for step_idx in range(self.max_steps):
            next_beam: list[_BeamEntry] = []

            for entry in beam:
                if entry.total_diff == 0:
                    next_beam.append(entry)
                    continue

                candidates = _generate_candidates(entry.states, targets, world=world)

                # Neural guidance: reorder candidates by predicted likelihood
                if features is not None and self.predictor._trained:
                    candidates = self.predictor.rank_candidates(candidates, features)

                # Try top candidates (limit to avoid explosion)
                tried = 0
                for cand in candidates:
                    if tried >= 80:  # cap per beam entry
                        break

                    new_states = []
                    success = True
                    for state in entry.states:
                        try:
                            result = _apply_candidate(cand, state)
                            if result is None or result.shape != state.shape:
                                success = False
                                break
                            new_states.append(result)
                        except Exception:
                            success = False
                            break

                    if not success:
                        continue

                    tried += 1
                    new_diff = sum(_pixel_diff(s, t) for s, t in zip(new_states, targets))
                    if new_diff >= entry.total_diff:
                        continue

                    new_step = ConstructionStep(
                        op=cand.op, args=cand.args,
                        description=cand.description,
                        diff_before=entry.total_diff,
                        diff_after=new_diff,
                    )
                    next_beam.append(_BeamEntry(
                        states=new_states,
                        steps=entry.steps + [new_step],
                        total_diff=new_diff,
                    ))

            if not next_beam:
                break

            next_beam.extend(e for e in beam if e.total_diff > 0)

            seen: dict[bytes, _BeamEntry] = {}
            for entry in next_beam:
                key = b"".join(s.tobytes() for s in entry.states)
                if key not in seen or entry.total_diff < seen[key].total_diff:
                    seen[key] = entry
            next_beam = sorted(seen.values())
            beam = next_beam[:self.beam_width]

            if beam[0].total_diff == 0:
                break

        best = min(beam, key=lambda e: e.total_diff)
        solved = best.total_diff == 0
        program = _build_program(best.steps) if solved else None

        return StepperResult(
            solved=solved,
            program=program,
            steps_description=tuple(s.description for s in best.steps),
            per_demo=tuple(
                ConstructionResult(
                    solved=_pixel_diff(s, t) == 0,
                    steps=tuple(best.steps),
                    final_diff=_pixel_diff(s, t),
                )
                for s, t in zip(best.states, targets)
            ),
        )

    def _record_solve(
        self,
        task_id: str,
        demos: tuple[DemoPair, ...],
        program: Program,
        step_ops: list[str],
    ) -> None:
        """Record a solve: store program + add training example."""
        self._solved[task_id] = program

        try:
            world = build_world_model(demos, task_id=task_id)
            features = extract_features(world)
        except Exception:
            return

        # Extract op names from step descriptions
        ops = []
        for desc in step_ops:
            # Extract the primary op name from description
            op = desc.split("(")[0].split("[")[0].strip()
            if op:
                ops.append(op)

        if ops:
            self.predictor.add_example(TrainingExample(
                task_id=task_id,
                features=features,
                winning_ops=tuple(ops),
            ))

    def _try_fitters(self, task_id: str, demos: tuple[DemoPair, ...]) -> Program | None:
        """Try graph-native fitters as a fallback."""
        from aria.sketch import SketchGraph
        from aria.sketch_fit import (
            fit_framed_periodic_repair, fit_composite_role_alignment,
            fit_canvas_construction, fit_object_movement, fit_grid_transform,
            specialize_sketch,
        )
        from aria.sketch_compile import compile_sketch_graph, CompileTaskProgram

        for fitter in [
            fit_framed_periodic_repair, fit_composite_role_alignment,
            fit_canvas_construction, fit_object_movement, fit_grid_transform,
        ]:
            try:
                sketch = fitter(demos, task_id=task_id)
                if sketch is None:
                    continue
                graph = SketchGraph.from_sketch(sketch)
                spec = specialize_sketch(graph, demos)
                result = compile_sketch_graph(graph, spec, demos)
                if isinstance(result, CompileTaskProgram):
                    vr = aria_verify(result.program, demos)
                    if vr.passed:
                        return result.program
            except Exception:
                pass
        return None
