"""Canonical data contract for the soccer experiments.

The contract is intentionally stricter than either upstream dataset.  Its most
important job is provenance enforcement:

* ``synloc_calibration_only`` may contain RGB metadata and camera calibration,
  but no athlete/world-coordinate annotations.
* ``gsr_annotation_backed`` may contain image boxes and pitch positions, and is
  therefore labelled as an external, annotation-backed validation track.

This prevents a future adapter refactor from silently turning the main
calibration-oracle experiment into an experiment that reads target answers from
``position_on_pitch``/``bbox_pitch``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


SCHEMA_VERSION = "soccer-spatial-frame-v1"
SYNLOC_CALIBRATION_TRACK = "synloc_calibration_only"
GSR_ANNOTATION_TRACK = "gsr_annotation_backed"
VALID_TRACKS = {SYNLOC_CALIBRATION_TRACK, GSR_ANNOTATION_TRACK}

# These names are forbidden anywhere in calibration-only source metadata.  The
# upstream COCO file can contain them, but the adapter must discard them before
# constructing a SoccerFrame.
_ANNOTATION_KEYS = {
    "annotations",
    "position_on_pitch",
    "bbox_pitch",
    "bbox_pitch_raw",
    "keypoints_3d",
    "world_position",
    "pitch_xy",
}


def _finite(values: Iterable[float]) -> bool:
    import math

    return all(math.isfinite(float(v)) for v in values)


def _matrix(value: Optional[Sequence[Sequence[float]]], rows: int, cols: int,
            name: str) -> Optional[Tuple[Tuple[float, ...], ...]]:
    if value is None:
        return None
    if len(value) != rows or any(len(row) != cols for row in value):
        raise ValueError(f"{name} must be {rows}x{cols}")
    result = tuple(tuple(float(v) for v in row) for row in value)
    if not _finite(v for row in result for v in row):
        raise ValueError(f"{name} contains a non-finite value")
    return result


def _contains_forbidden_key(value: Any) -> Optional[str]:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _ANNOTATION_KEYS:
                return str(key)
            found = _contains_forbidden_key(child)
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for child in value:
            found = _contains_forbidden_key(child)
            if found:
                return found
    return None


@dataclass(frozen=True)
class PitchSpec:
    """Metric soccer pitch with a centre origin.

    ``x`` runs goal-to-goal, ``y`` runs sideline-to-sideline and ``z`` points
    upward.  SynLoc and GSR coordinates are retained in metres; adapters must
    explicitly convert any source that uses a different convention.
    """

    length_m: float = 105.0
    width_m: float = 68.0
    origin: str = "center"
    axes: str = "x_goal_to_goal_y_sideline_z_up"

    def validate(self) -> None:
        if not (90.0 <= float(self.length_m) <= 120.0):
            raise ValueError(f"implausible soccer pitch length: {self.length_m}")
        if not (45.0 <= float(self.width_m) <= 90.0):
            raise ValueError(f"implausible soccer pitch width: {self.width_m}")
        if self.origin != "center":
            raise ValueError("soccer v1 contract requires a centre-origin pitch")
        if self.axes != "x_goal_to_goal_y_sideline_z_up":
            raise ValueError("unsupported soccer coordinate axes")

    def contains(self, xy: Sequence[float], margin_m: float = 0.0) -> bool:
        x, y = map(float, xy[:2])
        return (
            -self.length_m / 2.0 - margin_m <= x <= self.length_m / 2.0 + margin_m
            and -self.width_m / 2.0 - margin_m <= y <= self.width_m / 2.0 + margin_m
        )


@dataclass(frozen=True)
class CameraCalibration:
    """Either canonical OpenCV pinhole parameters or native SynLoc parameters."""

    model: str
    intrinsic: Optional[Tuple[Tuple[float, ...], ...]] = None
    rotation_world_to_camera: Optional[Tuple[Tuple[float, ...], ...]] = None
    translation_world_to_camera: Optional[Tuple[float, float, float]] = None
    distortion: Tuple[float, ...] = ()
    native_camera_matrix: Optional[Tuple[Tuple[float, ...], ...]] = None
    provider: str = "dataset"

    @classmethod
    def opencv_pinhole(
        cls,
        intrinsic: Sequence[Sequence[float]],
        rotation_world_to_camera: Sequence[Sequence[float]],
        translation_world_to_camera: Sequence[float],
        distortion: Sequence[float] = (),
        provider: str = "dataset",
    ) -> "CameraCalibration":
        t = tuple(float(v) for v in translation_world_to_camera)
        if len(t) != 3:
            raise ValueError("translation_world_to_camera must have 3 values")
        obj = cls(
            model="opencv_pinhole",
            intrinsic=_matrix(intrinsic, 3, 3, "intrinsic"),
            rotation_world_to_camera=_matrix(
                rotation_world_to_camera, 3, 3, "rotation_world_to_camera"
            ),
            translation_world_to_camera=t,
            distortion=tuple(float(v) for v in distortion),
            provider=provider,
        )
        obj.validate()
        return obj

    @classmethod
    def synloc_native(
        cls,
        camera_matrix: Sequence[Sequence[float]],
        dist_poly: Sequence[float],
        provider: str = "SpiideoSynLoc",
    ) -> "CameraCalibration":
        rows = tuple(tuple(float(v) for v in row) for row in camera_matrix)
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            raise ValueError("SynLoc camera_matrix must be a non-empty rectangular matrix")
        obj = cls(
            model="synloc_sskit",
            native_camera_matrix=rows,
            distortion=tuple(float(v) for v in dist_poly),
            provider=provider,
        )
        obj.validate()
        return obj

    def validate(self) -> None:
        if self.model == "opencv_pinhole":
            if self.intrinsic is None or self.rotation_world_to_camera is None:
                raise ValueError("opencv_pinhole requires K and R")
            if self.translation_world_to_camera is None:
                raise ValueError("opencv_pinhole requires t")
        elif self.model == "synloc_sskit":
            if self.native_camera_matrix is None:
                raise ValueError("synloc_sskit requires native_camera_matrix")
        else:
            raise ValueError(f"unsupported calibration model: {self.model}")
        vals: List[float] = list(self.distortion)
        for matrix in (self.intrinsic, self.rotation_world_to_camera, self.native_camera_matrix):
            if matrix:
                vals.extend(v for row in matrix for v in row)
        if self.translation_world_to_camera:
            vals.extend(self.translation_world_to_camera)
        if not _finite(vals):
            raise ValueError("calibration contains a non-finite value")


@dataclass(frozen=True)
class EntityObservation:
    """A visible GSR entity.  Never valid on the calibration-only track."""

    entity_id: str
    kind: str
    bbox_xywh: Tuple[float, float, float, float]
    pitch_xy_m: Tuple[float, float]
    track_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

    def validate(self, pitch: PitchSpec) -> None:
        if len(self.bbox_xywh) != 4 or not _finite(self.bbox_xywh):
            raise ValueError(f"invalid bbox for {self.entity_id}")
        if self.bbox_xywh[2] <= 0 or self.bbox_xywh[3] <= 0:
            raise ValueError(f"non-positive bbox for {self.entity_id}")
        if len(self.pitch_xy_m) != 2 or not _finite(self.pitch_xy_m):
            raise ValueError(f"invalid pitch position for {self.entity_id}")
        if not pitch.contains(self.pitch_xy_m, margin_m=3.0):
            raise ValueError(f"pitch position outside plausible bounds: {self.pitch_xy_m}")


@dataclass
class SoccerFrame:
    schema_version: str
    research_track: str
    dataset_id: str
    split: str
    scene_group_id: str
    frame_id: str
    image_path: str
    width: int
    height: int
    pitch: PitchSpec = field(default_factory=PitchSpec)
    calibration: Optional[CameraCalibration] = None
    observations: List[EntityObservation] = field(default_factory=list)
    source_metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def answer_source(self) -> str:
        if self.research_track == SYNLOC_CALIBRATION_TRACK:
            return "calibration_geometry"
        return "bbox_pitch_annotation"

    @property
    def uses_human_target_annotations(self) -> bool:
        """Whether target-domain human labels contribute to numeric answers."""

        return self.research_track == GSR_ANNOTATION_TRACK

    @property
    def external_test_only(self) -> bool:
        return self.research_track == GSR_ANNOTATION_TRACK

    @property
    def training_allowed(self) -> bool:
        return self.research_track == SYNLOC_CALIBRATION_TRACK

    def validate(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"expected schema_version={SCHEMA_VERSION}")
        if self.research_track not in VALID_TRACKS:
            raise ValueError(f"unknown research track: {self.research_track}")
        for name, value in {
            "dataset_id": self.dataset_id,
            "split": self.split,
            "scene_group_id": self.scene_group_id,
            "frame_id": self.frame_id,
            "image_path": self.image_path,
        }.items():
            if not str(value).strip():
                raise ValueError(f"{name} must be non-empty")
        if int(self.width) <= 0 or int(self.height) <= 0:
            raise ValueError("image dimensions must be positive")
        self.pitch.validate()
        if self.research_track == SYNLOC_CALIBRATION_TRACK:
            if self.calibration is None:
                raise ValueError("calibration-only track requires camera calibration")
            self.calibration.validate()
            if self.observations:
                raise ValueError(
                    "calibration-only track forbids entity/world-coordinate annotations"
                )
            key = _contains_forbidden_key(self.source_metadata)
            if key:
                raise ValueError(
                    f"calibration-only source_metadata leaks annotation key: {key}"
                )
        else:
            if not self.observations:
                raise ValueError("GSR annotation-backed frame has no entity observations")
            for obs in self.observations:
                obs.validate(self.pitch)

    def to_dict(self) -> Dict[str, Any]:
        self.validate()
        data = asdict(self)
        data["answer_source"] = self.answer_source
        data["uses_target_qa_labels"] = False
        data["uses_entity_world_annotations"] = (
            self.research_track == GSR_ANNOTATION_TRACK
        )
        data["uses_human_target_annotations"] = self.uses_human_target_annotations
        data["external_test_only"] = self.external_test_only
        data["training_allowed"] = self.training_allowed
        return data


def frame_from_dict(data: Dict[str, Any]) -> SoccerFrame:
    pitch = PitchSpec(**data.get("pitch", {}))
    raw_cal = data.get("calibration")
    calibration = None
    if raw_cal:
        calibration = CameraCalibration(
            model=raw_cal["model"],
            intrinsic=(tuple(tuple(row) for row in raw_cal["intrinsic"])
                       if raw_cal.get("intrinsic") else None),
            rotation_world_to_camera=(
                tuple(tuple(row) for row in raw_cal["rotation_world_to_camera"])
                if raw_cal.get("rotation_world_to_camera") else None
            ),
            translation_world_to_camera=(
                tuple(raw_cal["translation_world_to_camera"])
                if raw_cal.get("translation_world_to_camera") else None
            ),
            distortion=tuple(raw_cal.get("distortion", ())),
            native_camera_matrix=(
                tuple(tuple(row) for row in raw_cal["native_camera_matrix"])
                if raw_cal.get("native_camera_matrix") else None
            ),
            provider=raw_cal.get("provider", "dataset"),
        )
    observations = [
        EntityObservation(
            entity_id=str(item["entity_id"]),
            kind=str(item["kind"]),
            bbox_xywh=tuple(item["bbox_xywh"]),
            pitch_xy_m=tuple(item["pitch_xy_m"]),
            track_id=(str(item["track_id"]) if item.get("track_id") is not None else None),
            attributes=dict(item.get("attributes", {})),
        )
        for item in data.get("observations", [])
    ]
    frame = SoccerFrame(
        schema_version=data.get("schema_version", ""),
        research_track=data["research_track"],
        dataset_id=data["dataset_id"],
        split=data["split"],
        scene_group_id=str(data["scene_group_id"]),
        frame_id=str(data["frame_id"]),
        image_path=data["image_path"],
        width=int(data["width"]),
        height=int(data["height"]),
        pitch=pitch,
        calibration=calibration,
        observations=observations,
        source_metadata=dict(data.get("source_metadata", {})),
    )
    frame.validate()
    # These fields are derived from ``research_track`` rather than trusted
    # input.  If a serialized contract includes them, require an exact match
    # so a tampered GSR row cannot silently become training-eligible when it is
    # reloaded for a cross-manifest audit.
    declared_provenance = {
        "answer_source": frame.answer_source,
        "uses_target_qa_labels": False,
        "uses_entity_world_annotations": frame.research_track == GSR_ANNOTATION_TRACK,
        "uses_human_target_annotations": frame.uses_human_target_annotations,
        "external_test_only": frame.external_test_only,
        "training_allowed": frame.training_allowed,
    }
    for key, expected in declared_provenance.items():
        if key not in data:
            continue
        actual = data[key]
        mismatch = (
            actual is not expected if isinstance(expected, bool) else actual != expected
        )
        if mismatch:
            raise ValueError(
                f"serialized provenance mismatch for {key}: "
                f"expected {expected!r}, got {actual!r}"
            )
    return frame
