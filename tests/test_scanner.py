from file_organizer.scanner import (
    scan_existing_folder_examples,
    scan_existing_folder_names,
    scan_files,
)


def test_scan_files_default_only_returns_root_files(tmp_path):
    root_file = tmp_path / "loose.txt"
    root_file.write_text("loose")
    nested = tmp_path / "pictures"
    nested.mkdir()
    (nested / "photo.jpg").write_text("fake")

    records = scan_files(tmp_path, include_nested=False, include_hidden=False)

    assert [record.rel_path for record in records] == ["loose.txt"]


def test_scan_existing_folder_names_does_not_read_nested_files(tmp_path):
    folder = tmp_path / "pictures"
    folder.mkdir()
    (folder / "photo.jpg").write_text("fake")

    records = scan_existing_folder_names(tmp_path, include_hidden=False)

    assert [record.rel_path for record in records] == ["pictures"]
    assert records[0].kind == "folder"


def test_scan_existing_folder_examples_samples_per_top_level_folder(tmp_path):
    pictures = tmp_path / "pictures"
    pictures.mkdir()
    for index in range(5):
        (pictures / f"{index}.jpg").write_text("fake")

    records = scan_existing_folder_examples(
        tmp_path,
        include_hidden=False,
        sample_size=2,
    )

    assert [record.rel_path for record in records] == ["pictures/0.jpg", "pictures/1.jpg"]
