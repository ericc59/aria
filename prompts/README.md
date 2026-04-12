# Prompt Pack

Claude handoff prompts currently staged here:


Operational note:

- If Claude gets stuck, uncertain, or wants a second pass on code quality, it should use the Codex CLI for code review/help instead of immediately handing control back.
- After each prompt, use the codex CLI to do a code review of the changes. fix any bugs or logic errors.