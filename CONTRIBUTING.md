# Contributing

Thanks for helping improve `file-organizer`.

## Development Setup

```bash
uv sync
uv run ruff check .
uv run pytest
```

The test suite should pass without Ollama or downloaded local models.

## Issues

- Use the bug report form for reproducible failures.
- Use the feature request form for proposed behavior changes.
- Use GitHub Discussions for usage questions and early ideas.
- Do not include secrets, private paths, or sensitive filenames in public issues.

## Pull Requests

- Keep changes focused and small.
- Add or update tests for behavior changes.
- Run lint and tests before opening a PR.
- Describe user-visible behavior changes in the PR summary.
- Link related issues when applicable.
- Wait for CI to pass before requesting review.

## Local Models

Tests should not require Ollama or downloaded models. Keep model-dependent behavior behind small seams that can be tested with fakes.
