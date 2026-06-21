from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from file_organizer.cluster import (
    Assignment,
    EnrichedFile,
    assign_to_existing_folder,
    build_folder_profiles,
    cluster_unassigned,
)
from file_organizer.config import UnmatchedPolicy, load_settings
from file_organizer.models import OllamaModels
from file_organizer.planner import build_move_plan, execute_plan
from file_organizer.scanner import (
    FileRecord,
    metadata_text,
    scan_existing_folder_examples,
    scan_existing_folder_names,
    scan_files,
)

app = typer.Typer(
    help="Organize a folder using local Ollama embeddings and image understanding.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def organize(
    folder: Annotated[Path, typer.Argument(help="Folder to organize.")],
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Only print the organization plan. Do not move files.")
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Move files without prompting for confirmation.",
        ),
    ] = False,
    env_file: Annotated[
        Path | None, typer.Option("--env-file", help="Path to a .env file.")
    ] = None,
    include_nested: Annotated[
        bool,
        typer.Option(
            "--include-nested",
            help=(
                "Also analyze and move files already inside subfolders. "
                "By default, only root-level files are organized."
            ),
        ),
    ] = False,
    profile_subfolders: Annotated[
        bool,
        typer.Option(
            "--profile-subfolders",
            help=(
                "Analyze a sample of files inside existing subfolders as folder examples. "
                "By default, only top-level folder names are used."
            ),
        ),
    ] = False,
    profile_sample_size: Annotated[
        int,
        typer.Option(
            "--profile-sample-size",
            min=0,
            help=(
                "Files sampled per top-level folder with --profile-subfolders. "
                "Use 0 to profile all files."
            ),
        ),
    ] = 5,
    include_hidden: Annotated[
        bool, typer.Option("--include-hidden", help="Include hidden files and folders.")
    ] = False,
    allow_singleton_clusters: Annotated[
        bool,
        typer.Option(
            "--allow-singleton-clusters",
            help="Create a new folder even for one unmatched file.",
        ),
    ] = False,
    workers: Annotated[
        int,
        typer.Option(
            "--workers",
            min=1,
            help="Number of files to analyze in parallel.",
        ),
    ] = 2,
    unmatched_policy: Annotated[
        UnmatchedPolicy | None,
        typer.Option(
            "--unmatched-policy",
            case_sensitive=False,
            help="What to do with files that do not match a folder or cluster.",
        ),
    ] = None,
) -> None:
    settings = load_settings(env_file)
    policy = unmatched_policy or settings.unmatched_policy
    root = folder.expanduser().resolve()
    if not root.is_dir():
        raise typer.BadParameter(f"{root} is not a directory")

    models = OllamaModels(
        host=settings.ollama_host,
        embed_model=settings.embed_model,
        vision_model=settings.vision_model,
        naming_model=settings.naming_model,
    )

    target_records = scan_files(root, include_nested=include_nested, include_hidden=include_hidden)
    if profile_subfolders:
        sample_size = profile_sample_size or None
        example_records = scan_existing_folder_examples(
            root,
            include_hidden=include_hidden,
            sample_size=sample_size,
        )
    else:
        example_records = scan_existing_folder_names(root, include_hidden=include_hidden)
    target_paths = {record.path for record in target_records}
    seed_records = [record for record in example_records if record.path not in target_paths]

    if not target_records:
        console.print("[yellow]No files found to organize.[/yellow]")
        return

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        enrich_task = progress.add_task(
            "Analyzing loose files...",
            total=len(target_records),
        )
        enriched_targets = enrich_records(
            target_records,
            models=models,
            text_head_chars=settings.text_head_chars,
            progress=progress,
            task_id=enrich_task,
            workers=workers,
        )
        seed_task = progress.add_task(
            "Profiling existing folders...",
            total=len(seed_records),
        )
        enriched_seeds = enrich_records(
            seed_records,
            models=models,
            text_head_chars=settings.text_head_chars,
            progress=progress,
            task_id=seed_task,
            workers=workers,
        )

    profiles = build_folder_profiles(enriched_targets + enriched_seeds)

    assignments: list[Assignment] = []
    unassigned: list[EnrichedFile] = []
    for file in enriched_targets:
        existing = assign_to_existing_folder(
            file,
            profiles,
            threshold=settings.existing_folder_threshold,
            margin=settings.existing_folder_margin,
        )
        if existing is not None and existing.target_folder != file.record.top_folder:
            assignments.append(existing)
        elif existing is not None:
            assignments.append(
                Assignment(
                    file=file,
                    target_folder=file.record.top_folder,
                    reason="already in matching folder",
                    score=existing.score,
                )
            )
        else:
            unassigned.append(file)

    clusterable = [file for file in unassigned if file.record.kind != "binary"]
    unmatched = [file for file in unassigned if file.record.kind == "binary"]

    new_assignments, still_unmatched = cluster_unassigned(
        clusterable,
        threshold=settings.new_cluster_threshold,
        min_cluster_size=settings.min_cluster_size,
        allow_singletons=allow_singleton_clusters,
        existing_names=set(profiles),
        model_namer=models if settings.naming_model else None,
    )
    assignments.extend(new_assignments)
    unmatched.extend(still_unmatched)

    if unmatched:
        nested_unmatched = [file for file in unmatched if file.record.top_folder]
        loose_unmatched = [file for file in unmatched if not file.record.top_folder]
        assignments.extend(
            Assignment(
                file=file,
                target_folder=file.record.top_folder,
                reason="left in current folder",
            )
            for file in nested_unmatched
        )
        if policy == UnmatchedPolicy.ERROR:
            if not loose_unmatched:
                pass
            else:
                console.print(
                    "[red]Some loose files did not match any existing folder or new cluster:[/red]"
                )
            for file in loose_unmatched:
                console.print(f"  {file.record.rel_path}")
            if loose_unmatched:
                raise typer.Exit(code=2)
        if policy == UnmatchedPolicy.OTHER:
            assignments.extend(
                Assignment(
                    file=file,
                    target_folder=settings.other_folder,
                    reason="unmatched fallback",
                )
                for file in loose_unmatched
            )
        else:
            assignments.extend(
                Assignment(file=file, target_folder=None, reason="left unmatched")
                for file in loose_unmatched
            )

    move_plan = build_move_plan(root, assignments)
    print_assignments(assignments, move_count=len(move_plan))

    if not move_plan:
        console.print("[green]No moves needed.[/green]")
        return

    if dry_run:
        console.print("[yellow]Dry run only. No files were moved.[/yellow]")
        return

    if not yes and not typer.confirm(f"Move {len(move_plan)} file(s)?"):
        console.print("[yellow]Canceled. No files were moved.[/yellow]")
        return
    execute_plan(move_plan)
    console.print(f"[green]Moved {len(move_plan)} file(s).[/green]")


def enrich_records(
    records: list[FileRecord],
    *,
    models: OllamaModels,
    text_head_chars: int,
    progress: Progress | None = None,
    task_id: int | None = None,
    workers: int = 2,
) -> list[EnrichedFile]:
    if workers <= 1 or len(records) <= 1:
        enriched: list[EnrichedFile] = []
        for record in records:
            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"Analyzing {record.rel_path}")
            enriched.append(enrich_record(record, models=models, text_head_chars=text_head_chars))
            if progress is not None and task_id is not None:
                progress.advance(task_id)
        return enriched

    results: list[EnrichedFile | None] = [None] * len(records)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                enrich_record,
                record,
                models=models,
                text_head_chars=text_head_chars,
            ): index
            for index, record in enumerate(records)
        }
        for future in as_completed(futures):
            index = futures[future]
            results[index] = future.result()
            if progress is not None and task_id is not None:
                progress.update(task_id, description=f"Analyzed {records[index].rel_path}")
                progress.advance(task_id)

    return [result for result in results if result is not None]


def enrich_record(
    record: FileRecord,
    *,
    models: OllamaModels,
    text_head_chars: int,
) -> EnrichedFile:
    caption = None
    if record.kind == "image":
        try:
            caption = models.describe_image(record.path)
        except Exception:
            caption = None
    summary = metadata_text(record, text_head_chars=text_head_chars, image_caption=caption)
    embedding = models.embed(summary)
    return EnrichedFile(record=record, summary=summary, embedding=embedding)


def print_assignments(assignments: list[Assignment], *, move_count: int) -> None:
    table = Table(title=f"Organization Plan ({move_count} move(s))")
    table.add_column("File", overflow="fold")
    table.add_column("Target")
    table.add_column("Reason")
    for assignment in assignments:
        target = assignment.target_folder or "[dim](leave in place)[/dim]"
        table.add_row(assignment.file.record.rel_path, target, assignment.reason)
    console.print(table)


if __name__ == "__main__":
    app()
