"""Fast, model-free audits for canonical soccer frames and generated QA."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .contract import (
    GSR_ANNOTATION_TRACK,
    SYNLOC_CALIBRATION_TRACK,
    SoccerFrame,
    frame_from_dict,
)


def audit_frames(frames: Sequence[SoccerFrame]) -> Dict[str, Any]:
    keys = set()
    track_counts: Counter[str] = Counter()
    scene_counts: Counter[str] = Counter()
    for frame in frames:
        frame.validate()
        key = (frame.dataset_id, frame.split, frame.scene_group_id, frame.frame_id)
        if key in keys:
            raise AssertionError(f"duplicate soccer frame key: {key}")
        keys.add(key)
        track_counts[frame.research_track] += 1
        scene_counts[f"{frame.dataset_id}:{frame.split}:{frame.scene_group_id}"] += 1
    return {
        "n_frames": len(frames),
        "track_counts": dict(track_counts),
        "n_scene_groups": len(scene_counts),
        "scene_counts": dict(scene_counts),
    }


def audit_split_isolation(frames: Iterable[SoccerFrame]) -> Dict[str, List[str]]:
    """Reject a recoverable scene_group_id appearing in multiple splits."""

    scene_splits: Dict[Tuple[str, str], set] = defaultdict(set)
    for frame in frames:
        scene_splits[(frame.dataset_id, frame.scene_group_id)].add(frame.split)
    overlaps = {
        f"{dataset}:{scene}": sorted(splits)
        for (dataset, scene), splits in scene_splits.items()
        if len(splits) > 1
    }
    if overlaps:
        raise AssertionError(f"scene-group split leakage: {overlaps}")
    return {}


def audit_qa(rows: Sequence[Dict[str, Any]], max_mcq_share: float = 0.60,
             min_mcq_for_balance_check: int = 10) -> Dict[str, Any]:
    ids = set()
    categories: Counter[str] = Counter()
    tracks: Counter[str] = Counter()
    mcq_answers: Counter[str] = Counter()
    for row in rows:
        qa_id = str(row.get("qa_id", ""))
        if not qa_id:
            raise AssertionError("QA missing qa_id")
        if qa_id in ids:
            raise AssertionError(f"duplicate qa_id: {qa_id}")
        ids.add(qa_id)
        track = str(row.get("research_track", ""))
        source = str(row.get("answer_source", ""))
        annotated = row.get("uses_entity_world_annotations")
        human_annotated = row.get("uses_human_target_annotations")
        external_only = row.get("external_test_only")
        training_allowed = row.get("training_allowed")
        if track == SYNLOC_CALIBRATION_TRACK:
            if (source != "calibration_geometry"
                    or annotated is not False
                    or human_annotated is not False
                    or external_only is not False
                    or training_allowed is not True):
                raise AssertionError(f"calibration-only QA provenance violation: {qa_id}")
        elif track == GSR_ANNOTATION_TRACK:
            if (source != "bbox_pitch_annotation"
                    or annotated is not True
                    or human_annotated is not True
                    or external_only is not True
                    or training_allowed is not False):
                raise AssertionError(f"GSR QA provenance violation: {qa_id}")
        else:
            raise AssertionError(f"unknown QA research_track: {track}")
        if row.get("uses_target_qa_labels") is not False:
            raise AssertionError(f"QA must declare uses_target_qa_labels=false: {qa_id}")
        category = str(row.get("category", ""))
        if type(row.get("signed_axes_visible")) is not bool:
            raise AssertionError(f"QA has non-boolean signed_axes_visible: {qa_id}")
        signed_axes_visible = row.get("signed_axes_visible") is True
        signed_task = category in {"localization_2d_pitch", "pitch_region_signed_marker"}
        if signed_task and not signed_axes_visible:
            raise AssertionError(f"signed task without visible axes: {qa_id}")
        if track == GSR_ANNOTATION_TRACK and signed_axes_visible:
            raise AssertionError(f"natural GSR QA cannot claim an axis overlay: {qa_id}")
        categories[category] += 1
        tracks[track] += 1
        if category.startswith("relational_reasoning"):
            answer = str(row.get("answer", "")).strip().upper()
            if answer not in {"A", "B"}:
                raise AssertionError(f"invalid relational answer {answer!r}: {qa_id}")
            mcq_answers[answer] += 1
    n_mcq = sum(mcq_answers.values())
    if n_mcq >= min_mcq_for_balance_check:
        worst = max(mcq_answers.values()) / n_mcq
        if worst > max_mcq_share:
            raise AssertionError(
                f"relational answer-position bias {worst:.1%} > {max_mcq_share:.1%}: "
                f"{dict(mcq_answers)}"
            )
    return {
        "n_qa": len(rows),
        "category_counts": dict(categories),
        "track_counts": dict(tracks),
        "relational_answer_counts": dict(mcq_answers),
    }


def assert_training_eligible_qa(rows: Sequence[Dict[str, Any]]) -> None:
    """Hard gate for target-SFT/GRPO prompt builders."""

    for row in rows:
        if (row.get("research_track") != SYNLOC_CALIBRATION_TRACK
                or row.get("answer_source") != "calibration_geometry"
                or row.get("uses_target_qa_labels") is not False
                or row.get("uses_entity_world_annotations") is not False
                or row.get("training_allowed") is not True
                or row.get("external_test_only") is not False
                or row.get("uses_human_target_annotations") is not False):
            raise AssertionError(
                f"QA {row.get('qa_id', '<missing>')} is not eligible for soccer training"
            )


def audit_manifests(manifest_paths: Sequence[str]) -> Dict[str, Any]:
    """Audit scene/source/coordinate isolation across independently prepared splits."""

    if len(manifest_paths) < 2:
        raise ValueError("cross-manifest audit requires at least two manifests")
    frames: List[SoccerFrame] = []
    source_hash_splits: Dict[str, set] = defaultdict(set)
    coordinate_splits: Dict[Tuple[str, float, float], set] = defaultdict(set)
    coordinate_seeds: Dict[str, set] = defaultdict(set)
    loaded = []
    for raw_path in manifest_paths:
        path = Path(raw_path).resolve()
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        split = str(manifest.get("split", ""))
        if not split:
            raise AssertionError(f"manifest lacks split: {path}")
        source_hash = str(manifest.get("source_metadata_sha256", ""))
        if source_hash:
            source_hash_splits[source_hash].add(split)
        frames_path = path.parent / str(manifest.get("frames_file", "frames.jsonl"))
        with frames_path.open("r", encoding="utf-8") as handle:
            local_frames = [frame_from_dict(json.loads(line)) for line in handle if line.strip()]
        frames.extend(local_frames)
        marker_path = path.parent / str(
            manifest.get("marker_records_file", "marker_records.jsonl")
        )
        marker_count = 0
        if marker_path.is_file():
            with marker_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    marker_count += 1
                    xy = row["pitch_xy_m"]
                    dataset = str(row["dataset_id"])
                    coordinate_splits[(dataset, round(float(xy[0]), 8),
                                       round(float(xy[1]), 8))].add(split)
                    coordinate_seeds[split].add(int(row["coordinate_seed"]))
        loaded.append({
            "manifest": str(path), "split": split,
            "n_frames": len(local_frames), "n_marker_records": marker_count,
        })

    audit_split_isolation(frames)
    reused_sources = {
        digest: sorted(splits) for digest, splits in source_hash_splits.items()
        if len(splits) > 1
    }
    if reused_sources:
        raise AssertionError(f"same source metadata reused across splits: {reused_sources}")
    overlapping_coordinates = {
        f"{dataset}:{x:.8f},{y:.8f}": sorted(splits)
        for (dataset, x, y), splits in coordinate_splits.items()
        if len(splits) > 1
    }
    if overlapping_coordinates:
        preview = dict(list(overlapping_coordinates.items())[:10])
        raise AssertionError(f"marker coordinates overlap across splits: {preview}")
    shared_seeds = set.intersection(*coordinate_seeds.values()) if len(coordinate_seeds) > 1 else set()
    if shared_seeds:
        raise AssertionError(f"coordinate seeds reused across splits: {sorted(shared_seeds)}")
    return {
        "protocol": "soccer-cross-manifest-audit-v1",
        "manifests": loaded,
        "n_frames": len(frames),
        "n_unique_metric_coordinates": len(coordinate_splits),
        "coordinate_seeds_by_split": {
            split: sorted(seeds) for split, seeds in coordinate_seeds.items()
        },
        "scene_split_overlap": {},
        "source_hash_split_overlap": {},
        "coordinate_split_overlap": {},
        "passed": True,
    }
