# Contributing

Thanks for helping improve `file-organizer`.

## Development Setup

```bash
uv sync
uv run ruff check .
uv run pytest
```

## Pull Requests

- Keep changes focused and small.
- Add or update tests for behavior changes.
- Run lint and tests before opening a PR.
- Describe user-visible behavior changes in the PR summary.

## Local Models

Tests should not require Ollama or downloaded models. Keep model-dependent behavior behind small seams that can be tested with fakes.

