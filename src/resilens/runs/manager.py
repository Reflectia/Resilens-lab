"""Single entry point used by CLI and REST API to start and execute runs.

The REST API (T-16) calls submit() in the request and execute() later in a background worker,
one run at a time; it reads runs through `store`.
"""

import json
import time

from resilens.config.schema import RunConfig
from resilens.core.pipeline import Pipeline
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidator
from resilens.metrics.detection import compute_detection_metrics, validate_annotation_pair
from resilens.metrics.telemetry import ResourceSampler
from resilens.metrics.tracking import compute_tracking_metrics, export_mot_detections, export_mot_format
from resilens.policy.log import export_policy_log, policy_summary
from resilens.reports.generator import ReportGenerator
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore


class RunManager:
    def __init__(self, store: RunStore, recover_interrupted: bool = True) -> None:
        self.store = store
        if recover_interrupted:
            self.store.recover_interrupted()

    def submit(self, config: RunConfig) -> str:
        """Register a queued run without processing its frames."""
        return self.store.create_run(config).run_id

    def execute(self, run_id: str) -> None:
        """Process a stored run, then persist its metrics, report and final status."""
        self.store.update_status(run_id, RunStatus.RUNNING)
        try:
            self.store.save_metrics(run_id, self._process_video(run_id))
            ReportGenerator(self.store).run_report(run_id)
            self.store.update_status(run_id, RunStatus.COMPLETED)
        except Exception as error:
            self.store.update_status(run_id, RunStatus.FAILED, error=str(error) or type(error).__name__)

    def _process_video(self, run_id: str) -> dict[str, float]:
        config = self.store.load_config(run_id)
        self.store.verify_source(run_id)
        self.store.verify_annotations(run_id)
        video = VideoValidator(max_resolution=config.profile.max_resolution).validate(config.source)
        if config.annotations or config.tracking_annotations:
            if not config.annotations or not config.tracking_annotations:
                raise ValueError("Both COCO and MOT annotations are required")
            for annotation in (config.annotations, config.tracking_annotations):
                if not annotation.is_file():
                    raise FileNotFoundError(f"Annotation file not found: {annotation}")
            if float(config.mode.detector_params.get("conf_threshold", 0.3)) > 0.3:
                raise ValueError("Detector confidence threshold cannot exceed evaluation threshold 0.3")
            images = json.loads(config.annotations.read_text(encoding="utf-8"))["images"]
            if len(images) != video.frame_count or any(
                image["width"] != video.width or image["height"] != video.height for image in images
            ):
                raise ValueError("COCO annotations do not match video frame count or dimensions")
            validate_annotation_pair(config.annotations, config.tracking_annotations)
        pipeline = Pipeline.from_config(config)
        if pipeline is None:
            raise NotImplementedError("Pipeline is not implemented yet for mode A: Detector and Tracker are required")

        sampler = ResourceSampler()
        sampler.start()
        replay = VideoReplay(video, config.replay_mode)
        started = time.perf_counter()
        try:
            frames = pipeline.run(replay)
        finally:
            elapsed_s = time.perf_counter() - started
            resources = sampler.stop()

        if frames is None:
            raise NotImplementedError("Pipeline.run is not implemented yet")
        processed_frames = len(frames)
        if processed_frames == 0:
            raise ValueError("Run processed no video frames")
        run_dir = self.store.run_dir(run_id)
        export_mot_detections(frames, run_dir / "detections.txt")
        export_mot_format(frames, run_dir / "tracks.txt")
        quality_metrics = {}
        if config.annotations or config.tracking_annotations:
            quality_metrics = compute_detection_metrics(frames, config.annotations)
            quality_metrics.update(
                compute_tracking_metrics(run_dir / "tracks.txt", config.tracking_annotations, len(frames))
            )
            (run_dir / "quality_detections.json").write_text(
                json.dumps(
                    [
                        {"frame": f.index, "bbox": d.bbox, "score": d.score, "class_id": d.class_id}
                        for f in frames
                        for d in f.detections
                    ]
                ),
                encoding="utf-8",
            )
        policy_metrics = policy_summary(frames)
        if policy_metrics:
            export_policy_log(frames, config.profile.name, run_dir / "policy_log.csv")
        return {
            "processing_time_s": elapsed_s,
            "fps": processed_frames / elapsed_s,
            "peak_ram_mb": resources["peak_ram_mb"],
            "processed_frames": float(processed_frames),
            "frames_dropped": float(replay.frames_dropped),
            "detections_per_frame": sum(len(ctx.detections) for ctx in frames) / processed_frames,
            "tracks_per_frame": sum(len(ctx.tracks) for ctx in frames) / processed_frames,
            **policy_metrics,
            **quality_metrics,
        }

    def compare_with_previous(self, run_id: str) -> dict:
        raise NotImplementedError("Run comparisons are not implemented yet")
