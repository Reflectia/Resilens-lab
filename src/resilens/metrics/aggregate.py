"""Combine independent benchmark sequences using the underlying evaluators."""

import json
import tempfile
from pathlib import Path

import numpy as np

from resilens.core.contracts import Detection, FrameContext
from resilens.metrics.detection import compute_detection_metrics
from resilens.metrics.tracking import tracking_metrics_from_sequences, tracking_sequence
from resilens.runs.store import RunStore


def aggregate_runs(store: RunStore, run_ids: list[str]) -> dict[str, float]:
    combined = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "person"}]}
    frames = []
    sequences = []
    for run_id in run_ids:
        store.verify_annotations(run_id)
        config = store.load_config(run_id)
        if not config.annotations or not config.tracking_annotations:
            raise ValueError(f"Run {run_id} has no annotations")
        coco = json.loads(config.annotations.read_text(encoding="utf-8"))
        category = next(c["id"] for c in coco["categories"] if c["name"] == "person")
        offset = len(frames)
        image_map = {}
        for image in coco["images"]:
            new_id = len(combined["images"]) + 1
            image_map[image["id"]] = new_id
            combined["images"].append({**image, "id": new_id})
            frames.append(FrameContext(index=new_id - 1, timestamp_s=0, image=np.empty((0, 0, 3))))
        for ann in coco["annotations"]:
            if ann["category_id"] == category:
                combined["annotations"].append(
                    {
                        **ann,
                        "id": len(combined["annotations"]) + 1,
                        "image_id": image_map[ann["image_id"]],
                        "category_id": 1,
                    }
                )
        detections = json.loads((store.run_dir(run_id) / "quality_detections.json").read_text(encoding="utf-8"))
        for detection in detections:
            frames[offset + detection["frame"]].detections.append(
                Detection(tuple(detection["bbox"]), detection["score"], detection["class_id"])
            )
        sequences.append(
            tracking_sequence(store.run_dir(run_id) / "tracks.txt", config.tracking_annotations, len(coco["images"]))
        )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "combined.json"
        path.write_text(json.dumps(combined), encoding="utf-8")
        result = compute_detection_metrics(frames, path)
    result.update(tracking_metrics_from_sequences(sequences))
    return result
