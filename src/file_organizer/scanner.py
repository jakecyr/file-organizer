from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}

DOCUMENT_EXTENSIONS = {
    ".doc",
    ".docx",
    ".odp",
    ".ods",
    ".odt",
    ".pages",
    ".pdf",
    ".ppt",
    ".pptx",
    ".rtf",
}

SPREADSHEET_EXTENSIONS = {
    ".xls",
    ".xlsm",
    ".xlsx",
}

ARCHIVE_EXTENSIONS = {
    ".7z",
    ".gz",
    ".rar",
    ".tar",
    ".tgz",
    ".zip",
}

CERTIFICATE_EXTENSIONS = {
    ".cer",
    ".crt",
    ".der",
    ".p12",
    ".p8",
    ".pem",
}

PROVISIONING_EXTENSIONS = {
    ".mobileprovision",
    ".provisionprofile",
}

DIAGRAM_EXTENSIONS = {
    ".excalidraw",
    ".svg",
}

CONFIG_EXTENSIONS = {
    ".plist",
}

TEXT_EXTENSIONS = {
    ".cfg",
    ".conf",
    ".csv",
    ".css",
    ".env",
    ".html",
    ".ini",
    ".js",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rb",
    ".rst",
    ".sql",
    ".toml",
    ".ts",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "__pycache__",
    "node_modules",
}


@dataclass(frozen=True)
class FileRecord:
    path: Path
    root: Path
    top_folder: str | None
    kind: str
    rel_path: str

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def extension(self) -> str:
        return self.path.suffix.lower() or "[no extension]"

    @property
    def is_loose(self) -> bool:
        return self.top_folder is None


def classify_file(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in DOCUMENT_EXTENSIONS:
        return "document"
    if suffix in SPREADSHEET_EXTENSIONS:
        return "spreadsheet"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    if suffix in CERTIFICATE_EXTENSIONS:
        return "certificate"
    if suffix in PROVISIONING_EXTENSIONS:
        return "provisioning"
    if suffix in DIAGRAM_EXTENSIONS:
        return "diagram"
    if suffix in CONFIG_EXTENSIONS:
        return "config"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "binary"


def scan_files(root: Path, *, include_nested: bool, include_hidden: bool) -> list[FileRecord]:
    root = root.resolve()
    records: list[FileRecord] = []
    paths = root.rglob("*") if include_nested else root.iterdir()
    for path in paths:
        if not path.is_file():
            continue
        relative_parts = path.relative_to(root).parts
        if not include_hidden and any(part.startswith(".") for part in relative_parts):
            continue
        if any(part in SKIP_DIRS for part in relative_parts):
            continue
        top_folder = relative_parts[0] if len(relative_parts) > 1 else None
        if top_folder and not include_nested:
            continue
        records.append(
            FileRecord(
                path=path,
                root=root,
                top_folder=top_folder,
                kind=classify_file(path),
                rel_path=str(path.relative_to(root)),
            )
        )
    return sorted(records, key=lambda record: record.rel_path.lower())


def scan_existing_folder_names(root: Path, *, include_hidden: bool) -> list[FileRecord]:
    root = root.resolve()
    records: list[FileRecord] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        if child.name in SKIP_DIRS or (child.name.startswith(".") and not include_hidden):
            continue
        records.append(
            FileRecord(
                path=child,
                root=root,
                top_folder=child.name,
                kind="folder",
                rel_path=child.name,
            )
        )
    return records


def scan_existing_folder_examples(
    root: Path,
    *,
    include_hidden: bool,
    sample_size: int | None,
) -> list[FileRecord]:
    root = root.resolve()
    records: list[FileRecord] = []
    for child in sorted(root.iterdir(), key=lambda item: item.name.lower()):
        if not child.is_dir():
            continue
        if child.name in SKIP_DIRS or (child.name.startswith(".") and not include_hidden):
            continue
        for path in iter_folder_files(child, sample_size=sample_size):
            relative_parts = path.relative_to(root).parts
            if not include_hidden and any(part.startswith(".") for part in relative_parts):
                continue
            if any(part in SKIP_DIRS for part in relative_parts):
                continue
            records.append(
                FileRecord(
                    path=path,
                    root=root,
                    top_folder=relative_parts[0],
                    kind=classify_file(path),
                    rel_path=str(path.relative_to(root)),
                )
            )
    return records


def iter_folder_files(folder: Path, *, sample_size: int | None) -> Iterator[Path]:
    count = 0
    stack = [folder]
    while stack:
        current = stack.pop()
        dirs: list[Path] = []
        files: list[Path] = []
        for child in sorted(current.iterdir(), key=lambda item: item.name.lower()):
            if child.is_dir():
                dirs.append(child)
            elif child.is_file():
                files.append(child)
        stack.extend(reversed(dirs))
        for file in files:
            yield file
            count += 1
            if sample_size is not None and count >= sample_size:
                return


def read_text_head(path: Path, max_chars: int) -> str:
    raw = path.read_bytes()[: max_chars * 4]
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw.decode("utf-8", errors="replace")
    text = text.replace("\x00", " ")
    return " ".join(text[:max_chars].split())


def metadata_text(
    record: FileRecord,
    *,
    text_head_chars: int,
    image_caption: str | None = None,
) -> str:
    name_keywords = filename_keywords(record.path)
    parts = [
        f"Filename: {record.name}",
        f"Filename keywords: {name_keywords}",
        f"Relative path: {record.rel_path}",
        f"Extension: {record.extension}",
        f"File kind: {record.kind}",
    ]
    if name_keywords:
        repeats = 3 if record.kind == "image" else 2
        parts.append("Important filename keywords: " + " ".join([name_keywords] * repeats))
    if record.top_folder:
        parts.append(f"Current top-level folder: {record.top_folder}")
    if record.kind == "image" and image_caption:
        parts.append(f"Image description and visible text: {image_caption}")
    elif record.kind == "text":
        if is_sensitive_text_file(record.path):
            parts.append("Document head omitted: likely credential, secret, token, or config file.")
            return "\n".join(parts)
        head = read_text_head(record.path, text_head_chars)
        if looks_sensitive_text(head):
            parts.append("Document head omitted: likely credential, secret, token, or config file.")
            return "\n".join(parts)
        if head:
            parts.append(f"Document head: {head}")
    return "\n".join(parts)


def filename_keywords(path: Path) -> str:
    stem = path.stem.replace("_", " ").replace("-", " ")
    tokens = [match.group(0).lower() for match in re.finditer(r"[A-Za-z][A-Za-z0-9]+", stem)]
    return " ".join(tokens)


def is_sensitive_text_file(path: Path) -> bool:
    name = path.name.lower()
    suffix = path.suffix.lower()
    sensitive_suffixes = {
        ".env",
        ".key",
    } | CERTIFICATE_EXTENSIONS | PROVISIONING_EXTENSIONS
    if suffix in sensitive_suffixes:
        return True
    sensitive_markers = {
        "apikey",
        "api-key",
        "authkey",
        "credential",
        "firebase-adminsdk",
        "googleservice-info",
        "private-key",
        "secret",
        "service-account",
        "subscriptionkey",
        "token",
    }
    return any(marker in name for marker in sensitive_markers)


def looks_sensitive_text(text: str) -> bool:
    lowered = text.lower()
    sensitive_markers = {
        "-----begin private key-----",
        "api_key",
        "apikey",
        "auth_token",
        "client_x509_cert_url",
        "private_key",
        "private_key_id",
        "refresh_token",
        "service_account",
    }
    return any(marker in lowered for marker in sensitive_markers)
