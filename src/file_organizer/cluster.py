from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from file_organizer.scanner import FileRecord

STOPWORDS = {
    "about",
    "after",
    "and",
    "are",
    "description",
    "document",
    "extension",
    "file",
    "filename",
    "folder",
    "from",
    "image",
    "kind",
    "path",
    "relative",
    "screenshot",
    "text",
    "that",
    "the",
    "this",
    "visible",
    "with",
}

GENERIC_TOKENS = STOPWORDS | {
    "background",
    "black",
    "blue",
    "capture",
    "category",
    "computer",
    "current",
    "describes",
    "displaying",
    "does",
    "green",
    "header",
    "indicate",
    "indicates",
    "important",
    "jpeg",
    "jpg",
    "keywords",
    "level",
    "likely",
    "notes",
    "open",
    "page",
    "screen",
    "several",
    "show",
    "showing",
    "shown",
    "shows",
    "photo",
    "png",
    "red",
    "there",
    "txt",
    "type",
    "webpage",
    "white",
}


@dataclass
class EnrichedFile:
    record: FileRecord
    summary: str
    embedding: np.ndarray


@dataclass(frozen=True)
class Assignment:
    file: EnrichedFile
    target_folder: str | None
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class FolderProfile:
    name: str
    centroid: np.ndarray
    count: int
    tokens: set[str]


def build_folder_profiles(files: list[EnrichedFile]) -> dict[str, FolderProfile]:
    source_files = files
    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for file in source_files:
        if file.record.top_folder:
            grouped[file.record.top_folder].append(file.embedding)

    profiles: dict[str, FolderProfile] = {}
    for name, embeddings in grouped.items():
        grouped_files = [file for file in source_files if file.record.top_folder == name]
        centroid = normalize(np.mean(np.vstack(embeddings), axis=0))
        tokens = set().union(*(meaningful_tokens(file.summary) for file in grouped_files))
        profiles[name] = FolderProfile(
            name=name,
            centroid=centroid,
            count=len(embeddings),
            tokens=tokens,
        )
    return profiles


def assign_to_existing_folder(
    file: EnrichedFile,
    profiles: dict[str, FolderProfile],
    *,
    threshold: float,
    margin: float,
) -> Assignment | None:
    if not profiles:
        return None
    file_tokens = meaningful_tokens(file.summary)
    scored: list[tuple[str, float, float, int]] = []
    for name, profile in profiles.items():
        base_score = cosine(file.embedding, profile.centroid)
        overlap_count = len(file_tokens & profile.tokens)
        lexical_bonus = min(overlap_count, 3) * 0.05
        scored.append((name, base_score + lexical_bonus, base_score, overlap_count))
    scored.sort(key=lambda item: item[1], reverse=True)
    best_name, best_score, base_score, overlap_count = scored[0]
    if len(scored) == 1:
        if overlap_count < 2 and base_score < 0.85:
            return None
        return Assignment(
            file=file,
            target_folder=best_name,
            reason=(
                f"matched existing folder ({best_score:.2f}, "
                f"base {base_score:.2f}, lexical {overlap_count})"
            ),
            score=best_score,
        )

    runner_up_score = scored[1][1]
    score_gap = best_score - runner_up_score
    if best_score >= threshold and score_gap >= margin:
        lexical_note = f", lexical {overlap_count}" if overlap_count else ""
        return Assignment(
            file=file,
            target_folder=best_name,
            reason=(
                f"matched existing folder ({best_score:.2f}, "
                f"base {base_score:.2f}, gap {score_gap:.2f}{lexical_note})"
            ),
            score=best_score,
        )
    return None


def cluster_unassigned(
    files: list[EnrichedFile],
    *,
    threshold: float,
    min_cluster_size: int,
    allow_singletons: bool,
    existing_names: set[str],
    model_namer: object | None = None,
) -> tuple[list[Assignment], list[EnrichedFile]]:
    clusters: list[list[EnrichedFile]] = []
    centroids: list[np.ndarray] = []

    for file in files:
        best_index = None
        best_score = -1.0
        for index, centroid in enumerate(centroids):
            score = cosine(file.embedding, centroid)
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None and best_score >= threshold:
            clusters[best_index].append(file)
            centroids[best_index] = normalize(
                np.mean(np.vstack([item.embedding for item in clusters[best_index]]), axis=0)
            )
        else:
            clusters.append([file])
            centroids.append(file.embedding)

    assignments: list[Assignment] = []
    unmatched: list[EnrichedFile] = []
    used_names = set(existing_names)
    for cluster in clusters:
        if len(cluster) < min_cluster_size and not allow_singletons:
            unmatched.extend(cluster)
            continue
        folder_name = name_cluster(cluster, used_names, model_namer=model_namer)
        used_names.add(folder_name)
        for file in cluster:
            assignments.append(
                Assignment(file=file, target_folder=folder_name, reason="new semantic cluster")
            )
    return assignments, unmatched


def name_cluster(
    cluster: list[EnrichedFile],
    used_names: set[str],
    *,
    model_namer: object | None = None,
) -> str:
    if model_namer is not None and hasattr(model_namer, "name_cluster"):
        try:
            generated = model_namer.name_cluster([file.summary for file in cluster])
        except Exception:
            generated = None
        if generated:
            candidate = sanitize_folder_name(generated)
            if candidate:
                return unique_name(candidate, used_names)

    fallback_name = fallback_cluster_name(
        cluster, [file.record.extension.removeprefix(".") for file in cluster]
    )
    if fallback_name in {"Screenshots", "Images"}:
        return unique_name(fallback_name, used_names)

    words: list[str] = []
    extensions: list[str] = []
    for file in cluster:
        extensions.append(file.record.extension.removeprefix("."))
        words.extend(meaningful_tokens(Path(file.record.name).stem))
        words.extend(meaningful_tokens(file.summary))

    word_counts = Counter(words)
    if word_counts:
        base = " ".join(word.title() for word, _ in word_counts.most_common(3))
    else:
        base = fallback_name
    return unique_name(sanitize_folder_name(base) or "Organized Files", used_names)


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in re.finditer(r"[A-Za-z][A-Za-z0-9_-]+", text)]


def meaningful_tokens(text: str) -> set[str]:
    return {
        token.replace("_", "-")
        for token in tokenize(text)
        if token not in GENERIC_TOKENS and len(token) > 2
    }


def fallback_cluster_name(cluster: list[EnrichedFile], extensions: list[str]) -> str:
    names = " ".join(file.record.name.lower() for file in cluster)
    kinds = {file.record.kind for file in cluster}
    if "screenshot" in names:
        return "Screenshots"
    if kinds == {"image"}:
        return "Images"
    ext = Counter(extensions).most_common(1)[0][0]
    if ext == "pdf":
        return "PDFs"
    if ext:
        return f"{ext.upper()} Files"
    return "Organized Files"


def sanitize_folder_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9 _.-]+", "", name).strip(" ._-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned[:80]


def unique_name(base: str, used_names: set[str]) -> str:
    existing_by_lower = {name.lower(): name for name in used_names}
    existing = existing_by_lower.get(base.lower())
    if existing is not None:
        return existing
    if base not in used_names:
        return base
    index = 2
    while f"{base} {index}".lower() in existing_by_lower:
        index += 1
    return f"{base} {index}"


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(left, right))


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm
