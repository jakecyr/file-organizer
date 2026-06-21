import numpy as np
from typer.testing import CliRunner

from file_organizer.cli import app, enrich_records
from file_organizer.scanner import FileRecord

runner = CliRunner()


class FakeModels:
    def embed(self, text: str) -> np.ndarray:
        return np.array([float(len(text)), 1.0], dtype=np.float32)

    def describe_image(self, path):  # noqa: ANN001
        return f"image {path.name}"


def test_enrich_records_parallel_preserves_input_order(tmp_path):
    first = tmp_path / "b.txt"
    second = tmp_path / "a.txt"
    first.write_text("first")
    second.write_text("second")
    records = [
        FileRecord(first, tmp_path, None, "text", "b.txt"),
        FileRecord(second, tmp_path, None, "text", "a.txt"),
    ]

    enriched = enrich_records(
        records,
        models=FakeModels(),
        text_head_chars=100,
        workers=2,
    )

    assert [file.record.rel_path for file in enriched] == ["b.txt", "a.txt"]


def test_no_folder_defaults_to_current_directory(monkeypatch, tmp_path):
    class DummyModels:
        def __init__(self, **kwargs):  # noqa: ANN003
            pass

    seen_roots = []

    def fake_scan_files(root, *, include_nested, include_hidden):  # noqa: ANN001
        seen_roots.append(root)
        return []

    monkeypatch.setattr("file_organizer.cli.OllamaModels", DummyModels)
    monkeypatch.setattr("file_organizer.cli.scan_files", fake_scan_files)

    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, [])

    assert result.exit_code == 0
    assert seen_roots
    assert seen_roots[0] == tmp_path.resolve()
    assert "No files found to organize." in result.output
