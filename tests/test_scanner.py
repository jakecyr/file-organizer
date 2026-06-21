from file_organizer.scanner import (
    classify_file,
    metadata_text,
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


def test_classifies_common_non_text_file_types(tmp_path):
    assert classify_file(tmp_path / "contract.pdf") == "document"
    assert classify_file(tmp_path / "export.xlsx") == "spreadsheet"
    assert classify_file(tmp_path / "AuthKey_ABC.p8") == "certificate"
    assert classify_file(tmp_path / "App.mobileprovision") == "provisioning"
    assert classify_file(tmp_path / "archive.zip") == "archive"


def test_metadata_omits_sensitive_text_head(tmp_path):
    secret = tmp_path / "service.json"
    secret.write_text('{"private_key_id": "abc", "private_key": "secret"}')
    record = scan_files(tmp_path, include_nested=False, include_hidden=False)[0]

    summary = metadata_text(record, text_head_chars=1000)

    assert "private_key" not in summary
    assert "Document head omitted" in summary
