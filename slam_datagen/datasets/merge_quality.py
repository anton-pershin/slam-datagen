from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, ListConfig

from slam_datagen.personal_data import PersonalData, PersonalDataGenerator
from slam_datagen.utils.typing import NestedStrDict

SparseRecord = dict[str, str]

_IDENTIFIER_TYPES: tuple[str, ...] = ("name", "ssn")


@dataclass
class Chunk:
    format: str
    owner_id: str
    content: str


@dataclass
class ChunkRow:
    identifier_type: str
    identifier_value: str
    owner_id: str
    fields: SparseRecord


@dataclass
class DatasetSample:
    ground_truth: PersonalData
    provided_identifiers: dict[str, str]
    chunks: list[Chunk]


def build_merge_quality_dataset(
    generator: PersonalDataGenerator,
    cfg: DictConfig,
) -> list[DatasetSample]:
    rng = random.Random(cfg.random_seed)
    formats = list(cfg.chunk_formats)

    samples: list[DatasetSample] = []
    for record in generator.generate(n=cfg.dataset_size):
        sparse_record, flat_fields = _sparsify_record(record, cfg, rng)
        provided_identifiers = {
            "name": record.unique_identifiers["name"],
            "ssn": record.unique_identifiers["ssn"],
        }

        chunks = _build_chunks_for_record(
            source_record=record,
            flat_fields=flat_fields,
            generator=generator,
            cfg=cfg,
            rng=rng,
            formats=formats,
        )

        samples.append(
            DatasetSample(
                ground_truth=sparse_record,
                provided_identifiers=provided_identifiers,
                chunks=chunks,
            )
        )

    return samples


def write_merge_quality_dataset(
    samples: list[DatasetSample],
    output_file: str | Path,
) -> Path:
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(_serialize_sample(sample)) + "\n")

    return output_path


def _serialize_sample(sample: DatasetSample) -> dict[str, Any]:
    return {
        "ground_truth": {
            "unique_identifiers": sample.ground_truth.unique_identifiers,
            "attributes": sample.ground_truth.attributes,
        },
        "provided_identifiers": sample.provided_identifiers,
        "chunks": [asdict(chunk) for chunk in sample.chunks],
    }


def _build_chunks_for_record(
    source_record: PersonalData,
    flat_fields: SparseRecord,
    generator: PersonalDataGenerator,
    cfg: DictConfig,
    rng: random.Random,
    formats: list[str],
) -> list[Chunk]:
    chunks: list[Chunk] = []
    target_partitions = _partition_fields(flat_fields, formats, rng)

    markdown_rows = getattr(cfg, "markdown_distractor_rows", 3)
    markdown_chunk_count = max(1, getattr(cfg, "markdown_chunks_per_person", 1))
    base_distractors = cfg.distractor_chunks_per_format

    for fmt in formats:
        identifier_type = _identifier_type(fmt=fmt, rng=rng)
        identifier_value = _identifier_value(identifier_type, source_record)
        target_fields = dict(target_partitions.get(fmt, {}))

        if fmt == "markdown":
            chunks.extend(
                _build_markdown_chunks(
                    identifier_type=identifier_type,
                    target_fields=target_fields,
                    target_identifier_value=identifier_value,
                    generator=generator,
                    cfg=cfg,
                    rng=rng,
                    rows_per_chunk=max(1, markdown_rows),
                    chunk_count=markdown_chunk_count,
                )
            )
            continue

        rows: list[ChunkRow] = []
        if target_fields:
            rows.append(
                ChunkRow(
                    identifier_type=identifier_type,
                    identifier_value=identifier_value,
                    owner_id="target",
                    fields=target_fields,
                )
            )

        distractors = generator.generate(n=base_distractors)
        for distractor in distractors:
            _, distractor_fields = _sparsify_record(distractor, cfg, rng)
            distractor_partition = _partition_fields(distractor_fields, [fmt], rng)
            rows.append(
                ChunkRow(
                    identifier_type=identifier_type,
                    identifier_value=_identifier_value(identifier_type, distractor),
                    owner_id="distractor",
                    fields=dict(distractor_partition.get(fmt, {})),
                )
            )

        for row in rows:
            if fmt == "json":
                chunks.append(_build_json_chunk(row))
            elif fmt == "xml":
                chunks.append(_build_xml_chunk(row))

    rng.shuffle(chunks)
    return chunks


def _build_markdown_chunks(
    identifier_type: str,
    target_fields: SparseRecord,
    target_identifier_value: str,
    generator: PersonalDataGenerator,
    cfg: DictConfig,
    rng: random.Random,
    rows_per_chunk: int,
    chunk_count: int,
) -> list[Chunk]:
    chunks: list[Chunk] = []

    target_probability = getattr(cfg, "markdown_target_row_probability", 0.5)

    for _ in range(chunk_count):
        rows: list[ChunkRow] = []
        include_target = bool(target_fields) and rng.random() < target_probability
        if include_target:
            rows.append(
                ChunkRow(
                    identifier_type=identifier_type,
                    identifier_value=target_identifier_value,
                    owner_id="target",
                    fields=target_fields,
                )
            )

        rows.extend(
            _sample_markdown_distractors(
                identifier_type=identifier_type,
                generator=generator,
                cfg=cfg,
                rng=rng,
                count=rows_per_chunk,
            )
        )

        if rows:
            chunks.append(_build_markdown_chunk(rows, identifier_type))

    return chunks


def _sample_markdown_distractors(
    identifier_type: str,
    generator: PersonalDataGenerator,
    cfg: DictConfig,
    rng: random.Random,
    count: int,
) -> list[ChunkRow]:
    distractor_rows: list[ChunkRow] = []
    for persona in generator.generate(n=count):
        _, fields = _sparsify_record(persona, cfg, rng)
        identifier_value = _identifier_value(identifier_type, persona)
        row_fields = dict(fields)
        if identifier_value:
            row_fields.setdefault(identifier_type, identifier_value)
        distractor_rows.append(
            ChunkRow(
                identifier_type=identifier_type,
                identifier_value=identifier_value,
                owner_id="distractor",
                fields=row_fields,
            )
        )
    return distractor_rows


def _sparsify_record(
    record: PersonalData,
    cfg: DictConfig,
    rng: random.Random,
) -> tuple[PersonalData, SparseRecord]:
    flat_attrs = _flatten_attributes(record.attributes)
    if not flat_attrs:
        return record, {}

    range_cfg = getattr(cfg, "ground_truth_field_range", None)
    keep = _sample_field_count(range_cfg, len(flat_attrs), rng)
    selected_keys = rng.sample(list(flat_attrs.keys()), k=keep)
    sparse_flat = {key: flat_attrs[key] for key in selected_keys}
    sparse_attrs = _unflatten_attributes(sparse_flat)

    sparse_record = PersonalData(
        unique_identifiers=record.unique_identifiers.copy(),
        attributes=sparse_attrs,
    )
    return sparse_record, sparse_flat


def _sample_field_count(range_cfg: Any, available: int, rng: random.Random) -> int:
    if available == 0:
        return 0

    range_pair = _as_range_pair(range_cfg)
    if range_pair is None:
        target = int(range_cfg) if range_cfg is not None else available
        range_pair = (target, target)

    min_fields, max_fields = range_pair
    lower = max(1, min(min_fields, available))
    upper = max(lower, min(max_fields, available))
    return rng.randint(lower, upper)


def _as_range_pair(range_cfg: Any) -> tuple[int, int] | None:
    if isinstance(range_cfg, ListConfig):
        seq = list(range_cfg)
    elif isinstance(range_cfg, (list, tuple)):
        seq = list(range_cfg)
    else:
        return None

    if len(seq) != 2:
        return None
    return int(seq[0]), int(seq[1])


def _partition_fields(
    fields: SparseRecord,
    formats: list[str],
    rng: random.Random,
) -> dict[str, SparseRecord]:
    partitions: dict[str, SparseRecord] = {fmt: {} for fmt in formats}
    if not fields:
        return partitions

    keys = list(fields.keys())
    rng.shuffle(keys)
    for idx, key in enumerate(keys):
        fmt = formats[idx % len(formats)]
        partitions[fmt][key] = fields[key]
    return partitions


def _build_json_chunk(row: ChunkRow) -> Chunk:
    nested = _unflatten_attributes(_row_fields_with_identifier(row))
    payload = {
        "owner": row.owner_id,
        "identifier_type": row.identifier_type,
        "identifier_value": row.identifier_value,
        "data": nested,
    }
    return Chunk(
        format="json",
        owner_id=row.owner_id,
        content=json.dumps(payload, indent=2, ensure_ascii=False),
    )


def _build_xml_chunk(row: ChunkRow) -> Chunk:
    data = _unflatten_attributes(_row_fields_with_identifier(row))
    lines = ["<record>"]
    for key in sorted(data):
        lines.append(_dict_to_xml(key, data[key], indent=1))
    lines.append("</record>")
    return Chunk(
        format="xml",
        owner_id=row.owner_id,
        content="\n".join(lines),
    )


def _build_markdown_chunk(rows: list[ChunkRow], identifier_type: str) -> Chunk:
    row_payloads: list[dict[str, Any]] = []
    column_names: set[str] = set()
    for row in rows:
        fields = _sorted_dict(_row_fields_with_identifier(row))
        column_names.update(fields.keys())
        row_payloads.append(
            {
                "owner_id": row.owner_id,
                "fields": fields,
                "identifier_value": row.identifier_value,
            }
        )

    ordered_columns = [identifier_type] + [
        column for column in sorted(column_names) if column != identifier_type
    ]

    header = "| " + " | ".join(ordered_columns) + " |"
    separator = "| " + " | ".join(["---"] * len(ordered_columns)) + " |"
    lines = [header, separator]
    for payload in row_payloads:
        fields = payload["fields"]
        row_values = [fields.get(column, "") for column in ordered_columns]
        lines.append("| " + " | ".join(row_values) + " |")

    owner_states = {payload["owner_id"] for payload in row_payloads}
    if owner_states == {"target"}:
        owner_id = "target"
    elif owner_states == {"distractor"}:
        owner_id = "distractor"
    else:
        owner_id = "mixed"

    return Chunk(
        format="markdown",
        owner_id=owner_id,
        content="\n".join(lines),
    )


def _dict_to_xml(tag: str, value: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(value, dict):
        lines = [f"{prefix}<{tag}>"]
        for key in sorted(value):
            lines.append(_dict_to_xml(key, value[key], indent + 1))
        lines.append(f"{prefix}</{tag}>")
        return "\n".join(lines)
    if isinstance(value, list):
        raise ValueError("List values are not supported in XML chunks")
    return f"{prefix}<{tag}>{_escape_xml(str(value))}</{tag}>"


def _flatten_attributes(attributes: NestedStrDict | dict[str, Any]) -> SparseRecord:
    flat: SparseRecord = {}

    def _recurse(current: Any, path: str) -> None:
        if isinstance(current, dict):
            for key, value in current.items():
                next_path = f"{path}__{key}" if path else key
                _recurse(value, next_path)
        else:
            flat[path] = str(current)

    _recurse(attributes, "")
    return flat


def _unflatten_attributes(flat: SparseRecord) -> dict[str, NestedStrDict]:
    nested: dict[str, Any] = {}
    for path, value in flat.items():
        keys = path.split("__") if path else [path]
        cursor: dict[str, Any] = nested
        for key in keys[:-1]:
            cursor = cursor.setdefault(key, {})  # type: ignore[assignment]
        cursor[keys[-1]] = value
    return nested


def _sorted_dict(data: SparseRecord) -> SparseRecord:
    return {key: data[key] for key in sorted(data)} if data else {}


def _escape_xml(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _identifier_type(fmt: str, rng: random.Random) -> str:
    if fmt not in {"json", "xml", "markdown"}:
        raise ValueError(f"Unsupported format '{fmt}'")
    return rng.choice(_IDENTIFIER_TYPES)


def _identifier_value(identifier_type: str, record: PersonalData) -> str:
    if identifier_type in record.unique_identifiers:
        return record.unique_identifiers[identifier_type]
    if identifier_type == "email":
        return _extract_email(record) or ""
    return ""


def _extract_email(record: PersonalData) -> str | None:
    contacts = record.attributes.get("contacts")
    if isinstance(contacts, dict):
        email = contacts.get("email")
        if isinstance(email, str):
            return email
    return None


def _row_fields_with_identifier(row: ChunkRow) -> SparseRecord:
    result = dict(row.fields)
    if row.identifier_value:
        result.setdefault(row.identifier_type, row.identifier_value)
    return result
