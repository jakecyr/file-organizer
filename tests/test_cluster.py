import numpy as np

from file_organizer.cluster import (
    EnrichedFile,
    assign_to_existing_folder,
    build_folder_profiles,
    cluster_unassigned,
    name_cluster,
)
from file_organizer.scanner import FileRecord


def test_assigns_to_existing_folder(tmp_path):
    root = tmp_path
    receipts = root / "Receipts"
    receipts.mkdir()
    seed_path = receipts / "invoice.txt"
    seed_path.write_text("invoice")
    loose_path = root / "target.txt"
    loose_path.write_text("receipt")

    seed = EnrichedFile(
        record=FileRecord(seed_path, root, "Receipts", "text", "Receipts/invoice.txt"),
        summary="invoice receipt",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    target = EnrichedFile(
        record=FileRecord(loose_path, root, None, "text", "target.txt"),
        summary="store receipt",
        embedding=np.array([0.99, 0.01], dtype=np.float32),
    )

    profiles = build_folder_profiles([seed])
    assignment = assign_to_existing_folder(target, profiles, threshold=0.9, margin=0.05)

    assert assignment is not None
    assert assignment.target_folder == "Receipts"


def test_clusters_unassigned_files(tmp_path):
    root = tmp_path
    first_path = root / "beach.jpg"
    second_path = root / "ocean.png"
    first_path.write_text("fake")
    second_path.write_text("fake")
    files = [
        EnrichedFile(
            record=FileRecord(first_path, root, None, "image", "beach.jpg"),
            summary="beach vacation ocean",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        EnrichedFile(
            record=FileRecord(second_path, root, None, "image", "ocean.png"),
            summary="ocean waves vacation",
            embedding=np.array([0.95, 0.05], dtype=np.float32),
        ),
    ]

    assignments, unmatched = cluster_unassigned(
        files,
        threshold=0.8,
        min_cluster_size=2,
        allow_singletons=False,
        existing_names=set(),
    )

    assert not unmatched
    assert len(assignments) == 2
    assert assignments[0].target_folder == assignments[1].target_folder


def test_single_existing_folder_needs_strong_match(tmp_path):
    root = tmp_path
    receipts = root / "Receipts"
    receipts.mkdir()
    seed_path = receipts / "receipt.txt"
    seed_path.write_text("receipt")
    loose_path = root / "release-notes.txt"
    loose_path.write_text("release notes")

    seed = EnrichedFile(
        record=FileRecord(seed_path, root, "Receipts", "text", "Receipts/receipt.txt"),
        summary="coffee receipt invoice total paid",
        embedding=np.array([1.0, 0.0], dtype=np.float32),
    )
    unrelated = EnrichedFile(
        record=FileRecord(loose_path, root, None, "text", "release-notes.txt"),
        summary="release notes api dashboard",
        embedding=np.array([0.78, 0.22], dtype=np.float32),
    )

    profiles = build_folder_profiles([seed])
    assignment = assign_to_existing_folder(unrelated, profiles, threshold=0.68, margin=0.06)

    assert assignment is None


def test_name_cluster_ignores_generic_metadata_tokens(tmp_path):
    root = tmp_path
    first = root / "Screenshot 2026-03-08 at 12.45.44 PM.png"
    second = root / "Screenshot 2026-04-07 at 8.39.52 PM.png"
    first.write_text("fake")
    second.write_text("fake")
    cluster = [
        EnrichedFile(
            record=FileRecord(first, root, None, "image", first.name),
            summary=(
                "Filename keywords: screenshot png keywords important "
                "Extension: .png File kind: image"
            ),
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        EnrichedFile(
            record=FileRecord(second, root, None, "image", second.name),
            summary=(
                "Filename keywords: screenshot png keywords important "
                "Extension: .png File kind: image"
            ),
            embedding=np.array([0.99, 0.01], dtype=np.float32),
        ),
    ]

    assert name_cluster(cluster, used_names=set()) == "Screenshots"


def test_name_cluster_prefers_screenshots_over_generic_caption_words(tmp_path):
    root = tmp_path
    first = root / "Screenshot 2026-03-08 at 12.45.44 AM.png"
    second = root / "Screenshot 2026-04-07 at 8.39.52 PM.png"
    first.write_text("fake")
    second.write_text("fake")
    cluster = [
        EnrichedFile(
            record=FileRecord(first, root, None, "image", first.name),
            summary="The image shows a screen. There is text that indicates a webpage.",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        ),
        EnrichedFile(
            record=FileRecord(second, root, None, "image", second.name),
            summary="The screenshot shows there are several visible items.",
            embedding=np.array([0.99, 0.01], dtype=np.float32),
        ),
    ]

    assert name_cluster(cluster, used_names=set()) == "Screenshots"


def test_name_cluster_reuses_existing_folder_ignoring_case(tmp_path):
    root = tmp_path
    screenshot = root / "Screenshot 2026-04-07 at 8.39.52 PM.png"
    screenshot.write_text("fake")
    cluster = [
        EnrichedFile(
            record=FileRecord(screenshot, root, None, "image", screenshot.name),
            summary="The image shows there is text that indicates a webpage.",
            embedding=np.array([1.0, 0.0], dtype=np.float32),
        )
    ]

    assert name_cluster(cluster, used_names={"screenshots"}) == "screenshots"
