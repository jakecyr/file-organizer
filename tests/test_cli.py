import numpy as np

from file_organizer.cli import enrich_records
from file_organizer.scanner import FileRecord


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
