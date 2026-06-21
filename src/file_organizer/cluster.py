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
    "cert",
    "client",
    "computer",
    "current",
    "describes",
    "displaying",
    "does",
    "green",
    "group",
    "header",
    "indicate",
    "indicates",
    "important",
    "jpeg",
    "jpg",
    "json",
    "key",
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
    "targeted",
    "targetedkeywords",
    "there",
    "txt",
    "type",
    "url",
    "uri",
    "webpage",
    "white",
    "x509",
}

ENTITY_TOKEN_STOPWORDS = {
    "at",
    "by",
    "copy",
    "final",
    "for",
    "from",
    "new",
    "old",
    "the",
    "with",
}

ENTITY_PREFIX_STOPWORDS = {
    "authkey",
    "canva",
    "cert",
    "certificate",
    "copy",
    "developer",
    "developerid",
    "displaystatement",
    "distribution",
    "download",
    "downloaddocument",
    "fd",
    "google",
    "googleservice",
    "guided",
    "img",
    "mac",
    "notice",
    "payment",
    "screenshot",
    "screen",
    "smart",
    "subscriptionkey",
    "the",
    "training",
    "transactions",
    "translation",
    "untitled",
}

ENTITY_SECOND_TOKEN_STOPWORDS = {
    "access",
    "agent",
    "agreement",
    "application",
    "code",
    "development",
    "exit",
    "form",
    "guide",
    "icon",
    "info",
    "key",
    "logo",
    "memo",
    "misc",
    "partnership",
    "plan",
    "prod",
    "prompts",
    "referral",
    "repo",
    "request",
    "statement",
    "transition",
    "widget",
}

ENTITY_COMPANY_SUFFIXES = {
    "co",
    "corp",
    "inc",
    "llc",
    "ltd",
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


@dataclass(frozen=True)
class EntityCandidate:
    key: str
    label: str
    token_count: int
    rank: int


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


def assign_to_repeated_entity_groups(
    files: list[EnrichedFile],
    *,
    existing_names: set[str],
    min_group_size: int = 2,
) -> tuple[list[Assignment], list[EnrichedFile], set[str]]:
    candidates_by_file = [entity_candidates(file.record.name) for file in files]
    counts: Counter[str] = Counter()
    labels: dict[str, Counter[str]] = defaultdict(Counter)
    token_counts: dict[str, int] = {}
    for candidates in candidates_by_file:
        seen_file_keys: set[str] = set()
        for candidate in candidates:
            labels[candidate.key][candidate.label] += 1
            token_counts[candidate.key] = max(
                token_counts.get(candidate.key, 0),
                candidate.token_count,
            )
            if candidate.key in seen_file_keys:
                continue
            counts[candidate.key] += 1
            seen_file_keys.add(candidate.key)

    eligible_keys = {key for key, count in counts.items() if count >= min_group_size}
    if not eligible_keys:
        return [], files, set(existing_names)

    assignments: list[Assignment] = []
    remaining: list[EnrichedFile] = []
    used_names = set(existing_names)
    folder_names: dict[str, str] = {}

    for file, file_candidates in zip(files, candidates_by_file, strict=False):
        candidates = [
            candidate
            for candidate in file_candidates
            if candidate.key in eligible_keys
        ]
        if not candidates:
            remaining.append(file)
            continue

        candidate = max(
            candidates,
            key=lambda item: (counts[item.key], token_counts[item.key], -item.rank),
        )
        folder_name = folder_names.get(candidate.key)
        if folder_name is None:
            label = choose_entity_label(labels[candidate.key])
            folder_name = unique_name(label, used_names)
            used_names.add(folder_name)
            folder_names[candidate.key] = folder_name
        assignments.append(
            Assignment(
                file=file,
                target_folder=folder_name,
                reason=f"repeated name group ({counts[candidate.key]} files)",
            )
        )

    return assignments, remaining, used_names


def suggest_folder_name(file: EnrichedFile) -> str | None:
    name = file.record.name.lower()
    summary = file.summary.lower()
    extension = file.record.extension
    combined = f"{name} {summary}"

    if file.record.kind == "image":
        if "screenshot" in combined or name.startswith("screen shot"):
            return "Screenshots"
        if any(marker in combined for marker in ("asset", "brand", "icon", "logo", "removebg")):
            return "Design Assets"
        return "Images"

    if (
        "firebase-adminsdk" in combined
        or "googleservice-info" in combined
        or "private_key_id" in combined
        or "service account" in combined
        or (
            extension in {".json", ".plist"}
            and "document head omitted: likely credential" in summary
        )
    ):
        return "Service Account Keys"

    if (
        file.record.kind in {"certificate", "provisioning"}
        or extension in {".cer", ".crt", ".der", ".mobileprovision", ".p12", ".p8", ".pem"}
        or any(
            marker in name
            for marker in (
                "authkey",
                "certificate",
                "developerid",
                "distribution",
                "private key",
                "provision",
                "subscriptionkey",
            )
        )
    ):
        return "Developer Credentials"

    if file.record.kind == "archive":
        return "Archives"

    if file.record.kind == "diagram" or extension in {".excalidraw", ".svg"}:
        return "Design Assets"

    if "transcript" in name:
        return "Transcripts"

    if any(marker in name for marker in ("1099", "tax document", "tax form")):
        return "Tax Documents"

    if any(
        marker in name
        for marker in (
            "claim",
            "consolidated transaction",
            "displaystatement",
            "invoice",
            "payment",
            "quote",
            "receipt",
            "statement",
            "transactions",
        )
    ):
        return "Financial Records"

    if any(
        marker in name
        for marker in (
            "agreement",
            "contract",
            "docusign",
            "legal",
            "memo",
            "operating agreement",
            "withdrawal request",
        )
    ):
        return "Legal Documents"

    if file.record.kind == "document":
        return "Documents"

    if file.record.kind == "spreadsheet":
        return "Spreadsheets"

    return None


def entity_candidates(filename: str) -> list[EntityCandidate]:
    tokens = filename_entity_tokens(Path(filename).stem)
    candidates: list[EntityCandidate] = []
    rank = 0

    if (
        len(tokens) >= 3
        and tokens[2].lower() in ENTITY_COMPANY_SUFFIXES
        and is_allowed_two_token_entity(tokens[0], tokens[1])
    ):
        label = format_entity_label(tokens[:3])
        candidates.append(
            EntityCandidate(
                key=entity_key(tokens[:3]),
                label=label,
                token_count=3,
                rank=rank,
            )
        )
        rank += 1

    if len(tokens) >= 2 and is_allowed_two_token_entity(tokens[0], tokens[1]):
        label = format_entity_label(tokens[:2])
        candidates.append(
            EntityCandidate(
                key=entity_key(tokens[:2]),
                label=label,
                token_count=2,
                rank=rank,
            )
        )
        rank += 1

    if tokens and is_allowed_single_token_entity(tokens[0]):
        label = format_entity_label(tokens[:1])
        candidates.append(
            EntityCandidate(
                key=entity_key(tokens[:1]),
                label=label,
                token_count=1,
                rank=rank,
            )
        )

    return candidates


def cluster_entity_name(cluster: list[EnrichedFile]) -> str | None:
    entity_counts: Counter[str] = Counter()
    entity_labels: dict[str, Counter[str]] = defaultdict(Counter)
    generic_export_files = 0

    for file in cluster:
        candidates = [
            candidate
            for candidate in entity_candidates(file.record.name)
            if candidate.token_count >= 2
        ]
        if not candidates:
            if is_generic_export_filename(file.record.name):
                generic_export_files += 1
            continue
        seen_keys: set[str] = set()
        for candidate in candidates:
            entity_labels[candidate.key][candidate.label] += 1
            if candidate.key in seen_keys:
                continue
            entity_counts[candidate.key] += 1
            seen_keys.add(candidate.key)

    if not entity_counts:
        return None

    key, count = entity_counts.most_common(1)[0]
    if count >= 2 or (count == 1 and generic_export_files >= len(cluster) - 1):
        return choose_entity_label(entity_labels[key])
    return None


def is_generic_export_filename(filename: str) -> bool:
    return not meaningful_token_sequence(Path(filename).stem)


def filename_entity_tokens(stem: str) -> list[str]:
    cleaned = stem.replace("_", " ").replace("-", " ")
    raw_tokens = re.findall(r"[A-Za-z][A-Za-z0-9]*", cleaned)
    tokens: list[str] = []
    for token in raw_tokens:
        normalized = token.lower()
        if normalized in ENTITY_TOKEN_STOPWORDS:
            continue
        if re.fullmatch(r"[a-f0-9]{8,}", normalized):
            continue
        if normalized.startswith("w") and normalized[1:].isdigit():
            continue
        if normalized.endswith("sdk"):
            continue
        if len(normalized) <= 1:
            continue
        tokens.append(token)
    return tokens


def is_allowed_two_token_entity(first: str, second: str) -> bool:
    first_key = first.lower()
    second_key = second.lower()
    if first_key in ENTITY_PREFIX_STOPWORDS or second_key in ENTITY_SECOND_TOKEN_STOPWORDS:
        return False
    if first_key.isdigit() or second_key.isdigit():
        return False
    return True


def is_allowed_single_token_entity(token: str) -> bool:
    normalized = token.lower()
    if normalized in ENTITY_PREFIX_STOPWORDS:
        return False
    if normalized in GENERIC_TOKENS:
        return False
    if token.isupper() and len(token) <= 2:
        return False
    return True


def entity_key(tokens: list[str]) -> str:
    return "".join(re.sub(r"[^A-Za-z0-9]+", "", token).lower() for token in tokens)


def format_entity_label(tokens: list[str]) -> str:
    if len(tokens) == 1:
        token = tokens[0]
        if token.isupper() and len(token) <= 6:
            return f"{token} Files"
        return format_entity_token(token)
    return " ".join(format_entity_token(token) for token in tokens)


def choose_entity_label(label_counts: Counter[str]) -> str:
    return max(
        label_counts,
        key=lambda label: (
            label_counts[label],
            entity_label_quality(label),
            len(label),
        ),
    )


def entity_label_quality(label: str) -> int:
    if " " not in label and any(char.islower() for char in label) and any(
        char.isupper() for char in label[1:]
    ):
        return 3
    if " " in label:
        return 2
    return 1


def format_entity_token(token: str) -> str:
    if token.isupper() and len(token) <= 6:
        return token
    if token.lower() in ENTITY_COMPANY_SUFFIXES:
        return token.upper()
    if any(char.islower() for char in token) and any(char.isupper() for char in token[1:]):
        return token
    return token.title()


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

    prefix_name = common_filename_prefix_name(cluster)
    if prefix_name:
        return unique_name(prefix_name, used_names)

    entity_name = cluster_entity_name(cluster)
    if entity_name:
        return unique_name(entity_name, used_names)

    words: list[str] = []
    extensions: list[str] = []
    for file in cluster:
        extensions.append(file.record.extension.removeprefix("."))
        words.extend(meaningful_token_sequence(Path(file.record.name).stem))
        words.extend(meaningful_token_sequence(file.summary))

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


def meaningful_token_sequence(text: str) -> list[str]:
    return [
        token.replace("_", "-")
        for token in tokenize(text)
        if token not in GENERIC_TOKENS and len(token) > 2
    ]


def common_filename_prefix_name(cluster: list[EnrichedFile]) -> str | None:
    if len(cluster) < 2:
        return None
    sequences = [
        meaningful_token_sequence(Path(file.record.name).stem.replace("_", " "))
        for file in cluster
    ]
    if not sequences or any(not sequence for sequence in sequences):
        return None

    common: list[str] = []
    for tokens in zip(*sequences, strict=False):
        first = tokens[0]
        if all(token == first for token in tokens):
            common.append(first)
        else:
            break

    if not common:
        return None
    if len(common) == 1:
        token = common[0]
        label = token.upper() if len(token) <= 6 else token.title()
        return f"{label} Files"
    return " ".join(token.upper() if len(token) <= 4 else token.title() for token in common[:3])


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
