# file-organizer

Local-first CLI for organizing loose files into folders with Ollama embeddings, image captions, filenames, extensions, and document text previews.

Commands:

```bash
file-organizer ~/Downloads
fo ~/Downloads
```

By default, `fo` prints a plan and asks for confirmation before moving files. Use `--dry-run` when you only want to preview.

## Prerequisites

Install Python 3.11 or newer:

```bash
python3 --version
```

Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Install Ollama from https://ollama.com, then pull the default local models:

```bash
ollama pull nomic-embed-text
ollama pull moondream
```

Start Ollama if it is not already running:

```bash
ollama serve
```

## Install

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/file-organizer.git
cd file-organizer
cp .env.example .env
./install.sh
```

The installer uses `uv`, installs the global commands `file-organizer` and `fo`, and adds tab completion for bash, zsh, or fish. Restart your shell after install.

If your shell cannot find the command, add uv's tool bin directory to your shell profile:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Run

Preview an organization plan without moving files:

```bash
fo ~/Downloads --dry-run
```

By default, `fo` only analyzes root-level files and top-level folder names. It does not inspect every file inside existing subfolders.

Move files after confirmation:

```bash
fo ~/Downloads
```

Skip the confirmation prompt for scripts:

```bash
fo ~/Downloads --yes
```

Also consider files already inside subfolders:

```bash
fo ~/Downloads --include-nested
```

Use a small sample of files inside existing subfolders as examples for matching. The default sample is 5 files per top-level folder.

```bash
fo ~/Downloads --profile-subfolders
```

Profile more files per folder, or use `0` to profile every nested file:

```bash
fo ~/Downloads --profile-subfolders --profile-sample-size 25
fo ~/Downloads --profile-subfolders --profile-sample-size 0
```

Analyze files in parallel. The default is 2 workers.

```bash
fo ~/Downloads --workers 4
```

Choose how unmatched loose files are handled:

```bash
fo ~/Downloads --unmatched-policy other
fo ~/Downloads --unmatched-policy error
fo ~/Downloads --unmatched-policy leave
```

## Configuration

Settings live in `.env` and use the `FILE_ORGANIZER_` prefix.

Useful defaults:

```bash
OLLAMA_HOST=http://localhost:11434
FILE_ORGANIZER_EMBED_MODEL=nomic-embed-text
FILE_ORGANIZER_VISION_MODEL=moondream
FILE_ORGANIZER_UNMATCHED_POLICY=other
FILE_ORGANIZER_OTHER_FOLDER=Other
```

Higher-quality vision models can be slower but more accurate, for example `llama3.2-vision:11b`. Stronger embedding alternatives include `mxbai-embed-large`.

## Development

```bash
uv sync
uv run ruff check .
uv run pytest
uv run file-organizer --help
```

## License

MIT
