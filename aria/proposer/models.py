"""Concrete ProposerModel implementations.

Provides API-backed proposers (Anthropic, OpenAI-compatible) and a
mock proposer for testing. All implement the ProposerModel protocol:
    generate(prompt: str, n: int) -> list[str]
"""

from __future__ import annotations

import os
import re
from typing import Any


class AnthropicProposer:
    """Proposer backed by the Anthropic API (Claude)."""

    def __init__(
        self,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 1024,
        temperature: float = 0.8,
        api_key: str | None = None,
    ):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, n: int) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            try:
                resp = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = resp.content[0].text
                program = _extract_program(text)
                if program:
                    results.append(program)
            except Exception as e:
                # Log but don't crash — proposer failures are expected
                results.append(f"-- generation error: {e}")
        return results


class OpenAIProposer:
    """Proposer backed by any OpenAI-compatible API (OpenAI, vLLM, Ollama, etc.)."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        max_tokens: int = 1024,
        temperature: float = 0.8,
        api_key: str | None = None,
    ):
        from openai import OpenAI

        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            base_url=base_url,
        )
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def generate(self, prompt: str, n: int) -> list[str]:
        results: list[str] = []
        # Use n parameter for batch generation when supported
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                n=min(n, 16),  # Most APIs cap n
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            for choice in resp.choices:
                program = _extract_program(choice.message.content or "")
                if program:
                    results.append(program)
        except Exception as e:
            results.append(f"-- generation error: {e}")

        # If we need more, make additional calls
        while len(results) < n:
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    n=1,
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                )
                program = _extract_program(resp.choices[0].message.content or "")
                if program:
                    results.append(program)
                else:
                    break  # Stop if model keeps producing unparseable output
            except Exception:
                break

        return results[:n]


class ClaudeCodeProposer:
    """Proposer backed by the `claude` CLI (Claude Code).

    Uses `claude -p` with optional Bash tool access so the model can
    test its programs before returning them. No API key needed.
    """

    def __init__(
        self,
        model: str | None = None,
        max_tokens: int = 1024,
        use_tools: bool = True,
        max_turns: int = 6,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.use_tools = use_tools
        self.max_turns = max_turns

    def _call_once(self, prompt: str, prompt_path: str) -> str:
        """Single claude -p call. Returns raw text or error string."""
        import subprocess

        cmd = [
            "claude", "-p",
            "--output-format", "text",
            "--max-turns", str(self.max_turns),
        ]
        if self.use_tools:
            cmd.extend(["--allowedTools", "Bash(command)"])
        if self.model:
            cmd.extend(["--model", self.model])

        try:
            with open(prompt_path) as pf:
                proc = subprocess.run(
                    cmd,
                    stdin=pf,
                    capture_output=True,
                    text=True,
                    timeout=300,
                )
            if proc.returncode != 0:
                return f"-- claude error (rc={proc.returncode}): {proc.stderr[:300]}"
            if not proc.stdout.strip():
                return f"-- claude empty response: stderr={proc.stderr[:300]}"
            return proc.stdout
        except subprocess.TimeoutExpired:
            return "-- timeout"
        except Exception as e:
            return f"-- error: {e}"

    def generate(self, prompt: str, n: int) -> list[str]:
        import tempfile
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # Write prompt once, share across parallel calls
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as f:
            f.write(prompt)
            prompt_path = f.name

        # Run all n calls in parallel
        results: list[str] = []
        with ThreadPoolExecutor(max_workers=min(n, 4)) as pool:
            futures = [pool.submit(self._call_once, prompt, prompt_path) for _ in range(n)]
            for future in as_completed(futures):
                text = future.result()
                program = _extract_program(text)
                if program:
                    results.append(program)
                else:
                    results.append(f"-- unparseable: {text[:300]}")

        try:
            os.unlink(prompt_path)
        except OSError:
            pass

        return results


class MockProposer:
    """Returns pre-set programs for testing."""

    def __init__(self, programs: list[str] | None = None):
        self._programs = programs or []
        self._idx = 0

    def generate(self, prompt: str, n: int) -> list[str]:
        results: list[str] = []
        for _ in range(n):
            if self._idx < len(self._programs):
                results.append(self._programs[self._idx])
                self._idx += 1
            else:
                break
        return results


def _extract_program(text: str) -> str | None:
    """Extract a step-language program from model output.

    Handles cases where the model wraps the program in markdown code blocks
    or includes explanatory text before/after the program.
    """
    # Try to find a code block first
    code_block = re.search(r"```(?:\w*\n)?(.*?)```", text, re.DOTALL)
    if code_block:
        text = code_block.group(1).strip()

    # Look for program structure: lines starting with bind/assert/yield
    lines = text.strip().splitlines()
    program_lines: list[str] = []
    in_program = False

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("--"):
            if in_program:
                program_lines.append("")  # preserve blank lines within program
            continue

        if stripped.startswith(("bind ", "assert ", "yield ")):
            in_program = True
            program_lines.append(stripped)
        elif in_program and not stripped.startswith(("bind ", "assert ", "yield ")):
            # End of program block
            break

    # Clean up trailing blank lines
    while program_lines and not program_lines[-1]:
        program_lines.pop()

    if not program_lines:
        return None

    # Ensure there's a yield
    has_yield = any(l.startswith("yield ") for l in program_lines)
    if not has_yield:
        return None

    return "\n".join(program_lines)
