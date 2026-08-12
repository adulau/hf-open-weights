#!/usr/bin/env python3
"""
hf_open_weights.py

Build a local catalogue of Hugging Face models that have downloadable model
weight files and classify them by their declared license.

The script uses the Hugging Face Hub API (not HTML scraping), stores results in
SQLite, and can export CSV and JSONL.

"Open weight" is not an authoritative Hugging Face boolean. This script keeps
three concepts separate:

  1. Weight availability: does the repo contain recognizable model-weight files?
  2. License classification:
       - open-source
       - open-weight
       - restricted
       - unknown
  3. Gating: whether Hub access requires accepting conditions / authentication.

Default policy "open-weight" emits models classified as either open-source or
open-weight. Use "--policy public-weights" to catalogue every repo that exposes
weight files, even if the license is unknown/restricted, and inspect the
license fields yourself.

Training information is extracted from model-card Markdown. Dataset provenance
uses both structured Model Card metadata and Hugging Face dataset links found
in the card text. Missing information is reported as missing; it is never
guessed.

The catalogue also retains statistics-friendly publisher and geographic
metadata. ``organization`` falls back to the repository namespace, while
``countries`` only contains values explicitly declared in Model Card metadata.

Examples:

  pip install -U huggingface_hub

  # Initial catalogue
  python hf_open_weights.py \
      --db hf-open-weights.sqlite \
      --csv hf-open-weights.csv \
      --jsonl hf-open-weights.jsonl

  # Test on the first 500 models returned by the Hub
  python hf_open_weights.py --limit 500

  # Start with the models that have received the most likes (stars)
  python hf_open_weights.py --sort most-starred --limit 500

  # Strictly use conventional open-source/open-content licenses
  python hf_open_weights.py --policy strict

  # Include every model repository exposing recognizable weight files
  python hf_open_weights.py --policy public-weights

  # Incremental-ish scan: sorted by last modification and stop at this date
  python hf_open_weights.py --since 2026-08-01T00:00:00Z

Authentication:
  Set HF_TOKEN if you want the Hub client to use a token. This can improve
  rate limits and allow access to metadata/cards you are entitled to access.

Caveat:
  License classification is a technical filter, not legal advice. A repository
  can contain files under terms that differ from its Model Card metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor, Future
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

from huggingface_hub import HfApi, ModelCard
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError


# ---------------------------------------------------------------------------
# License policy
# ---------------------------------------------------------------------------

# Licenses that are conventionally open-source software licenses or open
# content licenses without NC/ND restrictions. The exact suitability of an
# individual license for model weights is still something a user should check.
OPEN_SOURCE_LICENSES = {
    "apache-2.0",
    "mit",
    "afl-3.0",
    "artistic-2.0",
    "bsl-1.0",
    "bsd",
    "bsd-2-clause",
    "bsd-3-clause",
    "bsd-3-clause-clear",
    "cc0-1.0",
    "cc-by-2.0",
    "cc-by-2.5",
    "cc-by-3.0",
    "cc-by-4.0",
    "cc-by-sa-3.0",
    "cc-by-sa-4.0",
    "ecl-2.0",
    "epl-1.0",
    "epl-2.0",
    "etalab-2.0",
    "eupl-1.1",
    "eupl-1.2",
    "agpl-3.0",
    "gpl",
    "gpl-2.0",
    "gpl-3.0",
    "lgpl",
    "lgpl-2.1",
    "lgpl-3.0",
    "isc",
    "lppl-1.3c",
    "ms-pl",
    "mpl-2.0",
    "ncsa",
    "osl-3.0",
    "postgresql",
    "unlicense",
    "wtfpl",
    "zlib",
}

# Common "open-weight" / source-available model licenses. These are deliberately
# kept separate from OPEN_SOURCE_LICENSES because some have acceptable-use or
# other restrictions and should not be labelled OSI-style "open source".
OPEN_WEIGHT_LICENSES = {
    "openrail",
    "bigscience-openrail-m",
    "creativeml-openrail-m",
    "bigscience-bloom-rail-1.0",
    "bigcode-openrail-m",
    "openrail++",
    "openmdw-1.0",
    "openmdw-1.1",
    "llama2",
    "llama3",
    "llama3.1",
    "llama3.2",
    "llama3.3",
    "llama4",
    "grok2-community",
    "gemma",
}

# Licenses that Hugging Face recognizes but which clearly signal research-only,
# non-commercial, or otherwise limited use. They can still be included with
# --policy public-weights.
RESTRICTED_LICENSES = {
    "h-research",
    "intel-research",
    "apple-amlr",
    "fair-noncommercial-research-license",
    "deepfloyd-if-license",
    "cc-by-nc-2.0",
    "cc-by-nc-3.0",
    "cc-by-nc-4.0",
    "cc-by-nd-4.0",
    "cc-by-nc-nd-3.0",
    "cc-by-nc-nd-4.0",
    "cc-by-nc-sa-2.0",
    "cc-by-nc-sa-3.0",
    "cc-by-nc-sa-4.0",
}

UNKNOWN_LICENSES = {"", "unknown", "other", "cc", "gfdl", "odc-by", "odbl", "pddl", "c-uda", "cdla-sharing-1.0", "cdla-permissive-1.0", "cdla-permissive-2.0", "ofl-1.1", "lgpl-lr", "apple-ascl"}


# ---------------------------------------------------------------------------
# File / model-card detection
# ---------------------------------------------------------------------------

WEIGHT_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".ggml",
    ".h5",
    ".hdf5",
    ".onnx",
    ".msgpack",
    ".tflite",
    ".keras",
    ".nemo",
    ".mlmodel",
    ".params",
    ".npz",
)

# Avoid counting common non-weight binary files as model weights.
WEIGHT_BASENAME_PATTERNS = (
    re.compile(r"(?:^|/)pytorch_model(?:-\d+-of-\d+)?\.bin$", re.I),
    re.compile(r"(?:^|/)tf_model(?:\.h5)?$", re.I),
    re.compile(r"(?:^|/)flax_model(?:\.msgpack)?$", re.I),
    re.compile(r"(?:^|/)model(?:-\d+-of-\d+)?\.safetensors$", re.I),
    re.compile(r"(?:^|/)adapter_model(?:\.safetensors|\.bin)$", re.I),
)

DATASET_LINK_RE = re.compile(
    r"https?://(?:www\.)?huggingface\.co/datasets/"
    r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
    re.I,
)

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")

TRAINING_HEADING_RE = re.compile(
    r"\b("
    r"train(?:ing)?|pre[- ]?train(?:ing)?|post[- ]?train(?:ing)?|"
    r"fine[- ]?tun(?:e|ing)|finetun(?:e|ing)|"
    r"hyper[- ]?parameters?|optimizer|optimisation|optimization|"
    r"training data|data used|training procedure|training details|"
    r"training infrastructure|training recipe"
    r")\b",
    re.I,
)


@dataclass
class CatalogueRecord:
    model_id: str
    url: str
    author: str | None
    organization: str | None
    organization_source: str | None
    countries: list[str]
    languages: list[str]
    tags: list[str]
    last_modified: str | None
    created_at: str | None
    downloads: int | None
    likes: int | None
    followers: int | None
    gated: str
    pipeline_tag: str | None
    library_name: str | None
    base_models: list[str]
    weight_files: list[str]
    weight_file_count: int
    license: str | None
    license_name: str | None
    license_link: str | None
    license_class: str
    datasets_declared: list[str]
    datasets_linked: list[str]
    datasets_all: list[str]
    training_info_status: str
    training_text: str | None
    card_fetch_error: str | None
    scanned_at: str


def eprint(*args: Any, **kwargs: Any) -> None:
    print(*args, file=sys.stderr, **kwargs)


def isoformat(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    return str(value)


def parse_since(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    dt = datetime.fromisoformat(normalized)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def card_data_to_dict(card_data: Any) -> dict[str, Any]:
    if card_data is None:
        return {}
    if isinstance(card_data, dict):
        return card_data
    to_dict = getattr(card_data, "to_dict", None)
    if callable(to_dict):
        try:
            value = to_dict()
            return value if isinstance(value, dict) else {}
        except Exception:
            return {}
    # Last-resort extraction for old/new client variants.
    result: dict[str, Any] = {}
    for key in (
        "license",
        "license_name",
        "license_link",
        "datasets",
        "base_model",
        "library_name",
        "language",
        "tags",
        "organization",
        "organisation",
        "country",
        "countries",
    ):
        if hasattr(card_data, key):
            result[key] = getattr(card_data, key)
    return result


def as_string_list(value: Any) -> list[str]:
    """Normalize Model Card metadata that may be a scalar/list/dict."""
    if value is None:
        return []
    if isinstance(value, str):
        v = value.strip()
        return [v] if v else []
    if isinstance(value, (list, tuple, set)):
        result: list[str] = []
        for item in value:
            result.extend(as_string_list(item))
        return dedupe(result)
    if isinstance(value, dict):
        # Common metadata shapes use id/name/path/type.
        for key in ("id", "path", "repo_id", "name", "type"):
            v = value.get(key)
            if isinstance(v, str) and v.strip():
                return [v.strip()]
        return []
    return [str(value)]


def dedupe(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def get_license(model: Any, metadata: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    license_id = metadata.get("license")
    if isinstance(license_id, list):
        license_id = license_id[0] if license_id else None
    if license_id:
        license_id = str(license_id).strip().lower()

    # Fallback to "license:..." model tag.
    if not license_id:
        for tag in getattr(model, "tags", None) or []:
            if isinstance(tag, str) and tag.lower().startswith("license:"):
                license_id = tag.split(":", 1)[1].strip().lower()
                break

    license_name = metadata.get("license_name")
    license_link = metadata.get("license_link")
    return (
        str(license_id) if license_id else None,
        str(license_name) if license_name else None,
        str(license_link) if license_link else None,
    )


def classify_license(license_id: str | None) -> str:
    lic = (license_id or "").strip().lower()
    if lic in OPEN_SOURCE_LICENSES:
        return "open-source"
    if lic in OPEN_WEIGHT_LICENSES:
        return "open-weight"
    if lic in RESTRICTED_LICENSES:
        return "restricted"
    return "unknown"


def policy_accepts(license_class: str, policy: str) -> bool:
    if policy == "strict":
        return license_class == "open-source"
    if policy == "open-weight":
        return license_class in {"open-source", "open-weight"}
    if policy == "public-weights":
        return True
    raise ValueError(f"Unknown policy: {policy}")


def looks_like_weight_file(filename: str) -> bool:
    lower = filename.lower()

    # Explicit known patterns.
    if any(p.search(filename) for p in WEIGHT_BASENAME_PATTERNS):
        return True

    if not lower.endswith(WEIGHT_SUFFIXES):
        return False

    # Exclude a few common artifacts that share broad binary extensions.
    basename = lower.rsplit("/", 1)[-1]
    deny = {
        "training_args.bin",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    }
    if basename in deny:
        return False

    return True


def model_weight_files(model: Any) -> list[str]:
    siblings = getattr(model, "siblings", None) or []
    result: list[str] = []
    for sibling in siblings:
        filename = getattr(sibling, "rfilename", None)
        if not filename and isinstance(sibling, dict):
            filename = sibling.get("rfilename")
        if filename and looks_like_weight_file(str(filename)):
            result.append(str(filename))
    return sorted(dedupe(result))


def extract_training_sections(markdown: str, max_chars: int) -> str | None:
    """
    Extract likely training-related Markdown sections.

    We retain Markdown because code blocks/tables/hyperparameters are useful.
    Sections are selected using heading names only; no LLM inference is used.
    """
    if not markdown:
        return None

    lines = markdown.splitlines()
    headings: list[tuple[int, int, str]] = []
    for idx, line in enumerate(lines):
        m = HEADING_RE.match(line)
        if m:
            headings.append((idx, len(m.group(1)), m.group(2).strip()))

    selected_ranges: list[tuple[int, int]] = []

    for pos, (start, level, title) in enumerate(headings):
        if not TRAINING_HEADING_RE.search(title):
            continue

        end = len(lines)
        for next_start, next_level, _ in headings[pos + 1 :]:
            if next_level <= level:
                end = next_start
                break

        # Avoid duplicating nested training headings already captured by a
        # selected parent section.
        if any(existing_start <= start < existing_end for existing_start, existing_end in selected_ranges):
            continue
        selected_ranges.append((start, end))

    if not selected_ranges:
        return None

    sections = ["\n".join(lines[start:end]).strip() for start, end in selected_ranges]
    text = "\n\n".join(s for s in sections if s)
    if not text:
        return None

    if len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n\n[truncated]"
    return text


def extract_dataset_links(markdown: str) -> list[str]:
    return sorted(dedupe(m.group(1).rstrip(").,;:#") for m in DATASET_LINK_RE.finditer(markdown or "")))


def datasets_from_metadata(metadata: dict[str, Any], model: Any) -> list[str]:
    datasets = as_string_list(metadata.get("datasets"))

    # Hub may expose dataset tags even when Model Card object shape differs.
    for tag in getattr(model, "tags", None) or []:
        if isinstance(tag, str) and tag.lower().startswith("dataset:"):
            datasets.append(tag.split(":", 1)[1].strip())

    return sorted(dedupe(datasets))


def normalize_base_models(metadata: dict[str, Any]) -> list[str]:
    return sorted(dedupe(as_string_list(metadata.get("base_model"))))


def publisher_metadata(
    metadata: dict[str, Any], model_id: str, author: str | None
) -> tuple[str | None, str | None, list[str], list[str], list[str]]:
    """Return publisher/geography fields without inventing missing facts.

    Organization and country are not standard required Model Card fields, but
    they are commonly present as custom YAML keys. A repository namespace is a
    useful grouping key for statistics, so it is the documented organization
    fallback. It must not, however, be interpreted as proof that the namespace
    is a legal organization. Countries never receive an inferred fallback.
    """
    declared_org = metadata.get("organization") or metadata.get("organisation")
    organizations = as_string_list(declared_org)
    if organizations:
        organization = organizations[0]
        organization_source = "model-card"
    else:
        namespace = author or (model_id.split("/", 1)[0] if "/" in model_id else None)
        organization = str(namespace) if namespace else None
        organization_source = "repository-namespace" if organization else None

    countries = as_string_list(metadata.get("countries"))
    countries.extend(as_string_list(metadata.get("country")))
    languages = as_string_list(metadata.get("language"))
    tags = as_string_list(metadata.get("tags"))
    return (
        organization,
        organization_source,
        sorted(dedupe(countries)),
        sorted(dedupe(languages)),
        sorted(dedupe(tags)),
    )


def gated_to_string(value: Any) -> str:
    if value is None or value is False:
        return "false"
    if value is True:
        return "true"
    return str(value)


def fetch_card(model_id: str, token: str | None, max_training_chars: int) -> tuple[dict[str, Any], list[str], str | None, str | None]:
    """
    Return:
      (card_metadata, linked_datasets, training_text, error)
    """
    try:
        card = ModelCard.load(model_id, token=token)
        metadata = card_data_to_dict(card.data)
        text = card.text or ""
        linked = extract_dataset_links(text)
        training_text = extract_training_sections(text, max_training_chars)
        return metadata, linked, training_text, None
    except (HfHubHTTPError, RepositoryNotFoundError, OSError, ValueError) as exc:
        return {}, [], None, f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        # Keep a failed model from aborting a million-repo crawl.
        return {}, [], None, f"{type(exc).__name__}: {exc}"


def make_record(
    model: Any,
    *,
    list_metadata: dict[str, Any],
    card_metadata: dict[str, Any],
    linked_datasets: list[str],
    training_text: str | None,
    card_error: str | None,
    weight_files: list[str],
) -> CatalogueRecord:
    # The full card metadata is authoritative over the abbreviated list result.
    metadata = dict(list_metadata)
    metadata.update({k: v for k, v in card_metadata.items() if v is not None})

    license_id, license_name, license_link = get_license(model, metadata)
    license_class = classify_license(license_id)

    declared_datasets = datasets_from_metadata(metadata, model)
    all_datasets = sorted(dedupe(declared_datasets + linked_datasets))

    if training_text and all_datasets:
        training_status = "training-text-and-datasets"
    elif training_text:
        training_status = "training-text-only"
    elif all_datasets:
        training_status = "datasets-only"
    else:
        training_status = "not-documented"

    model_id = str(getattr(model, "id", None) or getattr(model, "modelId", None))
    author = getattr(model, "author", None)
    if author is None and "/" in model_id:
        author = model_id.split("/", 1)[0]
    organization, organization_source, countries, languages, tags = publisher_metadata(
        metadata, model_id, str(author) if author else None
    )

    return CatalogueRecord(
        model_id=model_id,
        url=f"https://huggingface.co/{model_id}",
        author=str(author) if author else None,
        organization=organization,
        organization_source=organization_source,
        countries=countries,
        languages=languages,
        tags=tags,
        last_modified=isoformat(getattr(model, "last_modified", None)),
        created_at=isoformat(getattr(model, "created_at", None)),
        downloads=getattr(model, "downloads", None),
        likes=getattr(model, "likes", None),
        followers=getattr(model, "followers", None),
        gated=gated_to_string(getattr(model, "gated", None)),
        pipeline_tag=getattr(model, "pipeline_tag", None),
        library_name=(
            metadata.get("library_name")
            or getattr(model, "library_name", None)
        ),
        base_models=normalize_base_models(metadata),
        weight_files=weight_files,
        weight_file_count=len(weight_files),
        license=license_id,
        license_name=license_name,
        license_link=license_link,
        license_class=license_class,
        datasets_declared=declared_datasets,
        datasets_linked=linked_datasets,
        datasets_all=all_datasets,
        training_info_status=training_status,
        training_text=training_text,
        card_fetch_error=card_error,
        scanned_at=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    )


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS models (
    model_id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    author TEXT,
    organization TEXT,
    organization_source TEXT,
    countries_json TEXT NOT NULL DEFAULT '[]',
    languages_json TEXT NOT NULL DEFAULT '[]',
    tags_json TEXT NOT NULL DEFAULT '[]',
    last_modified TEXT,
    created_at TEXT,
    downloads INTEGER,
    likes INTEGER,
    followers INTEGER,
    gated TEXT NOT NULL,
    pipeline_tag TEXT,
    library_name TEXT,
    base_models_json TEXT NOT NULL,
    weight_files_json TEXT NOT NULL,
    weight_file_count INTEGER NOT NULL,
    license TEXT,
    license_name TEXT,
    license_link TEXT,
    license_class TEXT NOT NULL,
    datasets_declared_json TEXT NOT NULL,
    datasets_linked_json TEXT NOT NULL,
    datasets_all_json TEXT NOT NULL,
    training_info_status TEXT NOT NULL,
    training_text TEXT,
    card_fetch_error TEXT,
    scanned_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_models_license_class
    ON models(license_class);
CREATE INDEX IF NOT EXISTS idx_models_last_modified
    ON models(last_modified);
CREATE INDEX IF NOT EXISTS idx_models_gated
    ON models(gated);
"""


UPSERT = """
INSERT INTO models (
    model_id, url, author, organization, organization_source, countries_json,
    languages_json, tags_json, last_modified, created_at, downloads, likes,
    followers, gated,
    pipeline_tag, library_name, base_models_json, weight_files_json,
    weight_file_count, license, license_name, license_link, license_class,
    datasets_declared_json, datasets_linked_json, datasets_all_json,
    training_info_status, training_text, card_fetch_error, scanned_at
) VALUES (
    :model_id, :url, :author, :organization, :organization_source,
    :countries_json, :languages_json, :tags_json,
    :last_modified, :created_at, :downloads, :likes, :followers,
    :gated, :pipeline_tag, :library_name, :base_models_json,
    :weight_files_json, :weight_file_count, :license, :license_name,
    :license_link, :license_class, :datasets_declared_json,
    :datasets_linked_json, :datasets_all_json, :training_info_status,
    :training_text, :card_fetch_error, :scanned_at
)
ON CONFLICT(model_id) DO UPDATE SET
    url=excluded.url,
    author=excluded.author,
    organization=excluded.organization,
    organization_source=excluded.organization_source,
    countries_json=excluded.countries_json,
    languages_json=excluded.languages_json,
    tags_json=excluded.tags_json,
    last_modified=excluded.last_modified,
    created_at=excluded.created_at,
    downloads=excluded.downloads,
    likes=excluded.likes,
    followers=excluded.followers,
    gated=excluded.gated,
    pipeline_tag=excluded.pipeline_tag,
    library_name=excluded.library_name,
    base_models_json=excluded.base_models_json,
    weight_files_json=excluded.weight_files_json,
    weight_file_count=excluded.weight_file_count,
    license=excluded.license,
    license_name=excluded.license_name,
    license_link=excluded.license_link,
    license_class=excluded.license_class,
    datasets_declared_json=excluded.datasets_declared_json,
    datasets_linked_json=excluded.datasets_linked_json,
    datasets_all_json=excluded.datasets_all_json,
    training_info_status=excluded.training_info_status,
    training_text=excluded.training_text,
    card_fetch_error=excluded.card_fetch_error,
    scanned_at=excluded.scanned_at
"""


def init_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    # Add metadata columns to catalogues created by versions before these
    # fields existed. SQLite has no ADD COLUMN IF NOT EXISTS.
    existing = {row[1] for row in conn.execute("PRAGMA table_info(models)")}
    migrations = {
        "organization": "TEXT",
        "organization_source": "TEXT",
        "countries_json": "TEXT NOT NULL DEFAULT '[]'",
        "languages_json": "TEXT NOT NULL DEFAULT '[]'",
        "tags_json": "TEXT NOT NULL DEFAULT '[]'",
        "downloads": "INTEGER",
        "likes": "INTEGER",
        "followers": "INTEGER",
    }
    for column, declaration in migrations.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE models ADD COLUMN {column} {declaration}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_models_organization ON models(organization)"
    )
    conn.commit()
    return conn


def db_dict(record: CatalogueRecord) -> dict[str, Any]:
    d = asdict(record)
    d["countries_json"] = json.dumps(d.pop("countries"), ensure_ascii=False)
    d["languages_json"] = json.dumps(d.pop("languages"), ensure_ascii=False)
    d["tags_json"] = json.dumps(d.pop("tags"), ensure_ascii=False)
    d["base_models_json"] = json.dumps(d.pop("base_models"), ensure_ascii=False)
    d["weight_files_json"] = json.dumps(d.pop("weight_files"), ensure_ascii=False)
    d["datasets_declared_json"] = json.dumps(d.pop("datasets_declared"), ensure_ascii=False)
    d["datasets_linked_json"] = json.dumps(d.pop("datasets_linked"), ensure_ascii=False)
    d["datasets_all_json"] = json.dumps(d.pop("datasets_all"), ensure_ascii=False)
    return d


def upsert_record(conn: sqlite3.Connection, record: CatalogueRecord) -> None:
    conn.execute(UPSERT, db_dict(record))


def row_to_export_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "model_id": row["model_id"],
        "url": row["url"],
        "author": row["author"],
        "organization": row["organization"],
        "organization_source": row["organization_source"],
        "countries": json.loads(row["countries_json"]),
        "languages": json.loads(row["languages_json"]),
        "tags": json.loads(row["tags_json"]),
        "last_modified": row["last_modified"],
        "created_at": row["created_at"],
        "downloads": row["downloads"],
        "likes": row["likes"],
        "followers": row["followers"],
        "gated": row["gated"],
        "pipeline_tag": row["pipeline_tag"],
        "library_name": row["library_name"],
        "base_models": json.loads(row["base_models_json"]),
        "weight_files": json.loads(row["weight_files_json"]),
        "weight_file_count": row["weight_file_count"],
        "license": row["license"],
        "license_name": row["license_name"],
        "license_link": row["license_link"],
        "license_class": row["license_class"],
        "datasets_declared": json.loads(row["datasets_declared_json"]),
        "datasets_linked": json.loads(row["datasets_linked_json"]),
        "datasets_all": json.loads(row["datasets_all_json"]),
        "training_info_status": row["training_info_status"],
        "training_text": row["training_text"],
        "card_fetch_error": row["card_fetch_error"],
        "scanned_at": row["scanned_at"],
    }


EXPORT_FIELDS = [
    "model_id",
    "url",
    "author",
    "organization",
    "organization_source",
    "countries",
    "languages",
    "tags",
    "last_modified",
    "created_at",
    "downloads",
    "likes",
    "followers",
    "gated",
    "pipeline_tag",
    "library_name",
    "base_models",
    "weight_files",
    "weight_file_count",
    "license",
    "license_name",
    "license_link",
    "license_class",
    "datasets_declared",
    "datasets_linked",
    "datasets_all",
    "training_info_status",
    "training_text",
    "card_fetch_error",
    "scanned_at",
]


def export_catalogue(conn: sqlite3.Connection, csv_path: Path | None, jsonl_path: Path | None) -> None:
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM models ORDER BY model_id COLLATE NOCASE")

    csv_file = None
    jsonl_file = None
    writer = None

    try:
        if csv_path:
            csv_file = csv_path.open("w", encoding="utf-8", newline="")
            writer = csv.DictWriter(csv_file, fieldnames=EXPORT_FIELDS)
            writer.writeheader()

        if jsonl_path:
            jsonl_file = jsonl_path.open("w", encoding="utf-8")

        for row in rows:
            item = row_to_export_dict(row)

            if writer:
                csv_item = dict(item)
                for field in (
                    "countries",
                    "languages",
                    "tags",
                    "base_models",
                    "weight_files",
                    "datasets_declared",
                    "datasets_linked",
                    "datasets_all",
                ):
                    csv_item[field] = json.dumps(csv_item[field], ensure_ascii=False)
                writer.writerow(csv_item)

            if jsonl_file:
                jsonl_file.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if csv_file:
            csv_file.close()
        if jsonl_file:
            jsonl_file.close()


def catalogue_statistics(conn: sqlite3.Connection) -> dict[str, Any]:
    """Build deterministic aggregate counts from the complete local catalogue."""
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM models")
    dimensions: dict[str, dict[str, int]] = {
        name: {}
        for name in (
            "organization",
            "country",
            "language",
            "license_class",
            "license",
            "pipeline_tag",
            "library_name",
            "training_info_status",
            "gated",
        )
    }
    total = 0
    downloads = 0
    likes = 0
    followers = 0

    def add(dimension: str, value: Any) -> None:
        label = str(value).strip() if value is not None else ""
        label = label or "(missing)"
        bucket = dimensions[dimension]
        bucket[label] = bucket.get(label, 0) + 1

    for row in rows:
        total += 1
        downloads += row["downloads"] or 0
        likes += row["likes"] or 0
        followers += row["followers"] or 0
        for field in (
            "organization",
            "license_class",
            "license",
            "pipeline_tag",
            "library_name",
            "training_info_status",
            "gated",
        ):
            add(field, row[field])
        countries = json.loads(row["countries_json"])
        languages = json.loads(row["languages_json"])
        for country in countries or [None]:
            add("country", country)
        for language in languages or [None]:
            add("language", language)

    ordered_dimensions = {
        name: dict(sorted(values.items(), key=lambda item: (-item[1], item[0].casefold())))
        for name, values in dimensions.items()
    }
    return {
        "model_count": total,
        "downloads_total": downloads,
        "likes_total": likes,
        "followers_total": followers,
        "dimensions": ordered_dimensions,
    }


def export_statistics(conn: sqlite3.Connection, path: Path | None) -> None:
    if not path:
        return
    with path.open("w", encoding="utf-8") as stats_file:
        json.dump(catalogue_statistics(conn), stats_file, ensure_ascii=False, indent=2)
        stats_file.write("\n")


# ---------------------------------------------------------------------------
# Hub crawl
# ---------------------------------------------------------------------------

def model_last_modified(model: Any) -> datetime | None:
    value = getattr(model, "last_modified", None)
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        try:
            text = str(value)
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            dt = datetime.fromisoformat(text)
        except ValueError:
            return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def iter_candidates(
    api: HfApi,
    *,
    policy: str,
    limit: int | None,
    since: datetime | None,
    sort: str,
) -> Iterator[tuple[Any, dict[str, Any], list[str]]]:
    """
    Stream ModelInfo records. We request the repository file list, but defer
    loading Model Card data to ``fetch_card``.

    Asking ``list_models`` for ``cardData`` makes huggingface_hub construct a
    ModelCardData object for every search result. One malformed card can then
    abort the entire lazy result iterator (for example, ``eval_results``
    without ``model_name``). Tags already contain the license information
    needed for this initial filter, and candidate cards are fetched below with
    per-repository error handling, so parsing card data here is both redundant
    and less resilient.
    """
    hub_sort = {
        "last-modified": "lastModified",
        "most-starred": "likes",
    }[sort]
    models = api.list_models(
        sort=hub_sort,
        # Be explicit rather than relying on the Hub client's default. Some
        # huggingface_hub versions/API paths default to ascending order when a
        # sort field is supplied, which makes ``likes`` start at zero.
        direction=-1,
        limit=limit,
        full=True,
    )

    for model in models:
        if getattr(model, "private", False):
            continue

        if since is not None:
            modified = model_last_modified(model)
            # Only last-modified results are chronological. With another sort,
            # keep scanning because a newer model can follow an older one.
            if (
                sort == "last-modified"
                and modified is not None
                and modified < since
            ):
                break
            if modified is not None and modified < since:
                continue

        weight_files = model_weight_files(model)
        if not weight_files:
            continue

        metadata = card_data_to_dict(getattr(model, "card_data", None))
        license_id, _, _ = get_license(model, metadata)
        license_class = classify_license(license_id)

        if not policy_accepts(license_class, policy):
            continue

        yield model, metadata, weight_files


def crawl(args: argparse.Namespace) -> None:
    token = os.environ.get("HF_TOKEN") or None
    api = HfApi(token=token)
    db_path = Path(args.db)
    conn = init_db(db_path)
    since = parse_since(args.since)

    eprint(
        f"Scanning Hugging Face models: policy={args.policy}, sort={args.sort}, "
        f"workers={args.workers}, since={isoformat(since) if since else 'beginning'}"
    )

    scanned = 0
    candidates = 0
    stored = 0
    card_errors = 0
    started = time.monotonic()

    # Keep a bounded number of in-flight README requests.
    max_in_flight = max(args.workers * 3, args.workers)

    def finalize(
        model: Any,
        list_metadata: dict[str, Any],
        weight_files: list[str],
        future: Future[tuple[dict[str, Any], list[str], str | None, str | None]],
    ) -> None:
        nonlocal stored, card_errors
        card_metadata, linked_datasets, training_text, card_error = future.result()
        if card_error:
            card_errors += 1

        record = make_record(
            model,
            list_metadata=list_metadata,
            card_metadata=card_metadata,
            linked_datasets=linked_datasets,
            training_text=training_text,
            card_error=card_error,
            weight_files=weight_files,
        )
        upsert_record(conn, record)
        stored += 1

        if stored % args.commit_every == 0:
            conn.commit()

        if stored % args.progress_every == 0:
            elapsed = max(time.monotonic() - started, 0.001)
            eprint(
                f"stored={stored:,} card_errors={card_errors:,} "
                f"rate={stored/elapsed:.2f} models/s"
            )

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            in_flight: list[
                tuple[
                    Any,
                    dict[str, Any],
                    list[str],
                    Future[tuple[dict[str, Any], list[str], str | None, str | None]],
                ]
            ] = []

            for model, list_metadata, weight_files in iter_candidates(
                api,
                policy=args.policy,
                limit=args.limit,
                since=since,
                sort=args.sort,
            ):
                scanned += 1
                candidates += 1

                future = pool.submit(
                    fetch_card,
                    str(getattr(model, "id", None) or getattr(model, "modelId", None)),
                    token,
                    args.max_training_chars,
                )
                in_flight.append((model, list_metadata, weight_files, future))

                if len(in_flight) >= max_in_flight:
                    item = in_flight.pop(0)
                    finalize(*item)

            for item in in_flight:
                finalize(*item)

        conn.commit()

        csv_path = Path(args.csv) if args.csv else None
        jsonl_path = Path(args.jsonl) if args.jsonl else None
        export_catalogue(conn, csv_path, jsonl_path)
        stats_path = Path(args.stats) if args.stats else None
        export_statistics(conn, stats_path)

        total_db = conn.execute("SELECT COUNT(*) FROM models").fetchone()[0]
        elapsed = max(time.monotonic() - started, 0.001)

        eprint(
            f"Done. candidates={candidates:,}, stored_this_run={stored:,}, "
            f"card_errors={card_errors:,}, db_total={total_db:,}, "
            f"elapsed={elapsed:.1f}s"
        )
        eprint(f"SQLite: {db_path}")
        if csv_path:
            eprint(f"CSV:    {csv_path}")
        if jsonl_path:
            eprint(f"JSONL:  {jsonl_path}")
        if stats_path:
            eprint(f"Stats:  {stats_path}")

    except KeyboardInterrupt:
        eprint("\nInterrupted; committing records already collected.")
        conn.commit()
        raise
    finally:
        conn.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Catalogue Hugging Face open-weight models, training information, and datasets."
    )
    parser.add_argument(
        "--policy",
        choices=("strict", "open-weight", "public-weights"),
        default="open-weight",
        help=(
            "strict=open-source licenses only; "
            "open-weight=open-source + known open-weight licenses (default); "
            "public-weights=all repos containing recognizable weight files."
        ),
    )
    parser.add_argument(
        "--db",
        default="hf-open-weights.sqlite",
        help="SQLite catalogue path (default: %(default)s)",
    )
    parser.add_argument(
        "--csv",
        default="hf-open-weights.csv",
        help="CSV export path; use empty string to disable (default: %(default)s)",
    )
    parser.add_argument(
        "--jsonl",
        default="hf-open-weights.jsonl",
        help="JSONL export path; use empty string to disable (default: %(default)s)",
    )
    parser.add_argument(
        "--stats",
        default="hf-open-weights-stats.json",
        help="Aggregate statistics JSON path; use empty string to disable (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Concurrent model-card fetches (default: %(default)s)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit Hub models enumerated; useful for testing.",
    )
    parser.add_argument(
        "--sort",
        choices=("last-modified", "most-starred"),
        default="last-modified",
        help=(
            "Order models by latest modification (default) or start with the "
            "most-liked/starred models."
        ),
    )
    parser.add_argument(
        "--since",
        default=None,
        help=(
            "Only scan models modified at/after this ISO-8601 timestamp. "
            "With --sort last-modified, the crawler stops when it reaches "
            "older models."
        ),
    )
    parser.add_argument(
        "--max-training-chars",
        type=int,
        default=12000,
        help="Maximum training-card text stored per model (default: %(default)s)",
    )
    parser.add_argument(
        "--commit-every",
        type=int,
        default=100,
        help="Commit SQLite every N stored models (default: %(default)s)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100,
        help="Print progress every N stored models (default: %(default)s)",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.commit_every < 1:
        parser.error("--commit-every must be >= 1")
    if args.progress_every < 1:
        parser.error("--progress-every must be >= 1")
    if args.max_training_chars < 0:
        parser.error("--max-training-chars must be >= 0")

    crawl(args)


if __name__ == "__main__":
    main()
