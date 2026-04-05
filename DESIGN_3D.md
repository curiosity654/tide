# TIDE 3D Detection Redesign Proposal (KITTI / nuScenes / Waymo)

This document describes how to extend TIDE from 2D detection/segmentation evaluation to 3D detection for KITTI, nuScenes, and Waymo.

## 1) Current limitations

The current core is 2D-first:

- `tidecv/data.py`: 2D annotations and detections
- `tidecv/quantify.py`: matching + error attribution + reporting
- `tidecv/ap.py`: AP calculation
- `tidecv/datasets.py`: COCO/LVIS/Pascal/Cityscapes loaders

Direct 3D support is blocked by:

1. no canonical 3D box schema,
2. 2D IoU-centric matching,
3. no 3D-specific error taxonomy,
4. no coordinate-frame normalization across datasets.

## 2) Design principles

1. Preserve 2D behavior and APIs.
2. Keep 3D logic modular (new evaluator path, not many `if is_3d` branches).
3. Standardize data schema first, then metrics and errors.
4. Use dataset adapters for KITTI/nuScenes/Waymo format differences.

## 3) Proposed architecture

### 3.1 Data model layer

Extend `Data` to support `task_type` and 3D geometry metadata:

- `task_type`: `2d_det | 2d_seg | 3d_det`
- canonical 3D fields:
  - `center`: `[x, y, z]`
  - `size`: `[l, w, h]`
  - `yaw`: rotation around **z-axis in a right-handed z-up canonical frame**
  - `velocity` (optional)
  - `frame`: `camera | lidar | ego | global`

Add 3D APIs:

- `add_ground_truth_3d(...)`
- `add_detection_3d(...)`

Each dataset adapter converts native box conventions to the canonical schema.

### 3.2 Evaluator and metric layer

In `tidecv/quantify.py`, split evaluation into:

- `Evaluator2D` (existing path)
- `Evaluator3D` (new path)

`Evaluator3D` should be protocol-aware:

- KITTI: 3D AP / BEV AP / AOS (class-specific thresholds)
- nuScenes: mAP + NDS components
- Waymo: Level_1/2 + mAPH

This keeps matching rules and metric definitions decoupled from the common pipeline.

### 3.3 Error taxonomy layer

Add 3D error classes under `tidecv/errors/`:

- `DepthError`
- `OrientationError`
- `ScaleError`
- `VelocityError` (for temporal datasets)

Keep existing common classes (`ClsError`, `DuplicateError`, `BackgroundError`, `MissedError`) and extend TIDE-style error contribution analysis to 3D.

### 3.4 Dataset adapter layer

In `tidecv/datasets.py`, add independent adapters:

- `KITTI3D(...)`, `KITTI3DResult(...)`
- `NuScenes3D(...)`, `NuScenes3DResult(...)`
- `Waymo3D(...)`, `Waymo3DResult(...)`

Each adapter should only:

1. parse source annotations/predictions,
2. map to canonical 3D schema,
3. attach metadata (frame, class map, protocol version).

### 3.5 Plotting/reporting layer

Extend `tidecv/plotting.py` with 3D-oriented views:

- BEV error visualization,
- depth/yaw/scale histograms,
- protocol-specific score tables.

## 4) Minimal implementation roadmap

### Phase 1 (smallest end-to-end path)

1. Add canonical 3D data schema and APIs.
2. Add `Evaluator3D` skeleton + matching interface.
3. Ship KITTI-only minimal end-to-end flow first.

### Phase 2

1. Add nuScenes protocol integration.
2. Add Waymo protocol integration.
3. Expand 3D error attribution/reporting.

### Phase 3

1. Add regression fixtures/tests.
2. Unify configuration for thresholds/class maps/frame conventions.
3. Optimize matching performance.

## 5) API sketch

```python
from tidecv import TIDE, datasets

tide = TIDE()

gt = datasets.KITTI3D("/path/to/kitti", split="val")
pred = datasets.KITTI3DResult("/path/to/preds.json")

tide.evaluate_3d(
    gt=gt,
    preds=pred,
    protocol="kitti_3d",
    match_thresholds={"Car": 0.7, "Pedestrian": 0.5, "Cyclist": 0.5},  # 3D IoU or distance thresholds by protocol
)

tide.summarize()
tide.plot()
```

## 6) Backward compatibility strategy

1. Keep `evaluate(...)` unchanged for 2D.
2. Add `evaluate_3d(...)` as a separate API.
3. Place 3D logic in new modules where possible (e.g., `tidecv/eval3d.py`).
4. Prioritize protocol alignment with official dataset evaluation definitions, then map into a unified TIDE error view.

---

For the smallest-risk rollout, implement and validate **KITTI-only first**, then extend to nuScenes and Waymo on the same architecture.
