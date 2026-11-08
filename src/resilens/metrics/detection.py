"""Detection quality via pycocotools (annotated videos only)"""

import contextlib
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from resilens.core.contracts import FrameContext


def validate_annotation_pair(coco_path: Path, mot_path: Path) -> None:
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = [c["id"] for c in coco["categories"] if c["name"] == "person"]
    if len(categories) != 1:
        raise ValueError("COCO annotations must have one person category")
    coco_boxes = Counter(
        (int(a["image_id"]), *(float(v) for v in a["bbox"]))
        for a in coco["annotations"]
        if a["category_id"] == categories[0] and not a.get("iscrowd", 0)
    )
    mot_boxes = Counter()
    for line in mot_path.read_text(encoding="utf-8").splitlines():
        parts = line.split(",")
        if len(parts) < 9:
            raise ValueError("Invalid MOT annotation")
        if float(parts[6]) != 0 and int(float(parts[7])) == 1:
            mot_boxes[(int(parts[0]), *(float(v) for v in parts[2:6]))] += 1
    if not coco_boxes or coco_boxes != mot_boxes:
        raise ValueError("COCO and MOT person annotations do not match")


def compute_detection_metrics(frames: list[FrameContext], annotations: Path) -> dict[str, float]:
    data = json.loads(annotations.read_text(encoding="utf-8"))
    images = data["images"]
    image_ids = [image["id"] for image in images]
    frame_ids = [frame.index + 1 for frame in frames]
    if (
        len(set(image_ids)) != len(image_ids)
        or len(set(frame_ids)) != len(frame_ids)
        or not set(frame_ids) <= set(image_ids)
    ):
        raise ValueError("COCO annotation frames do not match video frames")
    categories = [c["id"] for c in data["categories"] if c["name"] == "person"]
    if len(categories) != 1:
        raise ValueError("COCO annotations must have one person category")
    predictions = []
    for frame in frames:
        for detection in frame.detections:
            if detection.class_id != 0 or detection.score < 0.3:
                continue
            x1, y1, x2, y2 = detection.bbox
            predictions.append(
                {
                    "image_id": frame.index + 1,
                    "category_id": categories[0],
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": detection.score,
                }
            )
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO(str(annotations))
        if predictions:
            dt = gt.loadRes(predictions)
        else:
            dt = COCO()
            dt.dataset = {"images": images, "categories": data["categories"], "annotations": []}
            dt.createIndex()
        evaluator = COCOeval(gt, dt, "bbox")
        evaluator.params.catIds = categories
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    precision = evaluator.eval["precision"]
    valid = precision[:, :, 0, 0, 2]
    valid = valid[valid >= 0]
    if not valid.size:
        raise ValueError("No person ground truth available for detection evaluation")
    # COCO mAP uses maxDets=100; fixed-threshold precision/recall include every detection.
    largest_frame = max((sum(d.class_id == 0 and d.score >= 0.3 for d in f.detections) for f in frames), default=0)
    fixed = evaluator
    if largest_frame > 100:
        with contextlib.redirect_stdout(io.StringIO()):
            fixed = COCOeval(gt, dt, "bbox")
            fixed.params.catIds = categories
            fixed.params.maxDets = [1, 10, largest_frame]
            fixed.evaluate()
    evaluated = [x for x in fixed.evalImgs if x and x["aRng"] == fixed.params.areaRng[0]]
    tp = sum(int(np.count_nonzero((e["dtMatches"][0] > 0) & ~e["dtIgnore"][0])) for e in evaluated)
    fp = sum(int(np.count_nonzero((e["dtMatches"][0] == 0) & ~e["dtIgnore"][0])) for e in evaluated)
    gt_count = sum(int(np.count_nonzero(~e["gtIgnore"])) for e in evaluated)
    return {
        "map_50_95": float(valid.mean()),
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / gt_count if gt_count else 0.0,
        "detection_confidence_threshold": 0.3,
    }
