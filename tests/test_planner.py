import numpy as np

from file_organizer.cluster import Assignment, EnrichedFile
from file_organizer.planner import build_move_plan
from file_organizer.scanner import FileRecord


def test_build_move_plan_does_not_rename_file_already_in_target(tmp_path):
    root = tmp_path
    folder = root / "Projects"
    folder.mkdir()
    source = folder / "release-plan.txt"
    source.write_text("release plan")

    record = FileRecord(
        path=source,
        root=root,
        top_folder="Projects",
        kind="text",
        rel_path="Projects/release-plan.txt",
    )
    assignment = Assignment(
        file=EnrichedFile(record=record, summary="release plan", embedding=np.array([1.0])),
        target_folder="Projects",
        reason="already in matching folder",
    )

    assert build_move_plan(root, [assignment]) == []
