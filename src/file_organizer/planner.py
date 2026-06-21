from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from file_organizer.cluster import Assignment


@dataclass(frozen=True)
class MoveOperation:
    source: Path
    destination: Path
    reason: str


def build_move_plan(root: Path, assignments: list[Assignment]) -> list[MoveOperation]:
    operations: list[MoveOperation] = []
    for assignment in assignments:
        if assignment.target_folder is None:
            continue
        source = assignment.file.record.path
        destination_dir = root / assignment.target_folder
        desired_destination = destination_dir / source.name
        if source.resolve() == desired_destination.resolve():
            continue
        destination = unique_destination(desired_destination)
        operations.append(
            MoveOperation(source=source, destination=destination, reason=assignment.reason)
        )
    return operations


def execute_plan(operations: list[MoveOperation]) -> None:
    for operation in operations:
        operation.destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(operation.source), str(operation.destination))


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    parent = path.parent
    index = 2
    while True:
        candidate = parent / f"{stem} {index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1
