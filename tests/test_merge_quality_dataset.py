from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any

from omegaconf import OmegaConf

from slam_datagen.datasets.merge_quality import (
    Chunk,
    _IDENTIFIER_TYPES,
    _flatten_attributes,
    build_merge_quality_dataset,
)
from slam_datagen.personal_data import PersonalDataGenerator


def test_markdown_chunks_include_unique_identifier() -> None:
    cfg = OmegaConf.create(
        {
            "random_seed": 42,
            "dataset_size": 2,
            "chunk_formats": ["markdown"],
            "distractor_chunks_per_format": 0,
            "markdown_distractor_rows": 2,
            "markdown_chunks_per_person": 1,
            "markdown_target_row_probability": 1.0,
            "ground_truth_field_range": [5, 5],
        }
    )

    generator = PersonalDataGenerator(seed=42)
    samples = build_merge_quality_dataset(generator=generator, cfg=cfg)

    assert samples

    for sample in samples:
        target_identifiers = sample.provided_identifiers
        for chunk in sample.chunks:
            assert chunk.format == "markdown"
            first_cells = re.findall(r"^\| ([^|]+?) \|", chunk.content, flags=re.MULTILINE)
            identifier_column = first_cells[0].strip()
            assert identifier_column in _IDENTIFIER_TYPES
            assert target_identifiers[identifier_column] in chunk.content


def test_chunks_partition_ground_truth_attributes() -> None:
    cfg = OmegaConf.create(
        {
            "random_seed": 1337,
            "dataset_size": 100,
            "chunk_formats": ["json", "xml", "markdown"],
            "distractor_chunks_per_format": 1,
            "markdown_distractor_rows": 1,
            "markdown_chunks_per_person": 1,
            "markdown_target_row_probability": 0.5,
            "ground_truth_field_range": [3, 5],
        }
    )
#    cfg = OmegaConf.create(
#        {
#            "random_seed": 7,
#            "dataset_size": 1,
#            "chunk_formats": ["json", "xml", "markdown"],
#            "distractor_chunks_per_format": 1,
#            "ground_truth_field_range": [4, 4],
#            "markdown_chunks_per_person": 1,
#            "markdown_target_row_probability": 1.0,
#            "markdown_distractor_rows": 1,
#        }
#    )

    generator = PersonalDataGenerator(seed=7)
    samples = build_merge_quality_dataset(generator=generator, cfg=cfg)
    assert len(samples) == cfg.dataset_size

    for sample in samples:
        expected_attributes = _exclude_identifier_fields(
            _flatten_attributes(sample.ground_truth.attributes)
        )

        target_identifiers = sample.provided_identifiers

        chunk_attributes: dict[str, str] = {}
        for chunk in sample.chunks:
            if chunk.owner_id not in {"target", "mixed"}:
                continue
            chunk_attributes.update(_extract_chunk_attributes(chunk, target_identifiers))

        assert chunk_attributes == expected_attributes


def _exclude_identifier_fields(flat: dict[str, str]) -> dict[str, str]:
    return {key: value for key, value in flat.items() if key and key not in _IDENTIFIER_TYPES}


def _extract_chunk_attributes(chunk: Chunk, identifiers: dict[str, str]) -> dict[str, str]:
    if chunk.format == "json":
        payload = json.loads(chunk.content)
        flat = _flatten_attributes(payload["data"])
        return _exclude_identifier_fields(flat)
    if chunk.format == "xml":
        flat = _flatten_attributes(_xml_record_to_dict(chunk.content))
        return _exclude_identifier_fields(flat)
    if chunk.format == "markdown":
        return _collect_markdown_target_fields(chunk, identifiers)
    raise ValueError(f"Unsupported format '{chunk.format}' in test helper")


def _collect_markdown_target_fields(chunk: Chunk, identifiers: dict[str, str]) -> dict[str, str]:
    raw_lines = [line.rstrip() for line in chunk.content.splitlines() if line.strip()]
    if len(raw_lines) < 3:
        return {}

    columns = [cell.strip() for cell in raw_lines[0].strip("|").split("|")]
    identifier_column = columns[0]

    flattened: dict[str, str] = {}
    for raw_row in _coalesce_markdown_rows(raw_lines[2:]):
        cells = [cell.strip() for cell in raw_row.strip("|").split("|")]
        record = {columns[idx]: cells[idx] if idx < len(cells) else "" for idx in range(len(columns))}

        owner_id = record.get("owner_id", "") or "mixed"
        if owner_id not in {"target", "mixed"}:
            continue

        identifier_value = record.get(identifier_column, "")
        is_target_row = owner_id == "target" or identifier_value == identifiers.get(identifier_column, "")
        if not is_target_row:
            continue

        for key, value in record.items():
            if key in {"owner_id", identifier_column} or not value:
                continue
            flattened[key] = value
        if identifier_value:
            flattened.setdefault(identifier_column, identifier_value)

    return _exclude_identifier_fields(flattened)


def _coalesce_markdown_rows(lines: list[str]) -> list[str]:
    rows: list[str] = []
    current: list[str] = []
    for line in lines:
        if line.lstrip().startswith("|"):
            if current:
                rows.append("\n".join(current))
            current = [line]
        elif current:
            current.append(line)
    if current:
        rows.append("\n".join(current))
    return rows


def _xml_record_to_dict(content: str) -> dict[str, Any]:
    root = ET.fromstring(content)
    if root.tag != "record":
        raise ValueError("XML chunk root must be <record>")
    result: dict[str, Any] = {}
    for child in root:
        result[child.tag] = _xml_node_to_value(child)
    return result


def _xml_node_to_value(node: ET.Element) -> Any:
    if list(node):
        return {child.tag: _xml_node_to_value(child) for child in node}
    return node.text or ""
