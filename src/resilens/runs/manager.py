"""Single entry point used by CLI and REST API to start and execute runs.

The REST API (T-16) calls submit() in the request and execute() later in a background worker,
one run at a time; it reads runs through `store`.
"""

import time

from resilens.config.schema import RunConfig
from resilens.core.pipeline import Pipeline
from resilens.io.replay import VideoReplay
from resilens.io.validation import VideoValidator
from resilens.metrics.telemetry import ResourceSampler
from resilens.metrics.tracking import export_mot_detections, export_mot_format
from resilens.reports.generator import ReportGenerator
from resilens.runs.models import RunStatus
from resilens.runs.store import RunStore


class RunManager:
    def __init__(self, store: RunStore) -> None:
        self.store = store
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
        video = VideoValidator(max_resolution=config.profile.max_resolution).validate(config.source)
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
        return {
            "processing_time_s": elapsed_s,
            "fps": processed_frames / elapsed_s,
            "peak_ram_mb": resources["peak_ram_mb"],
            "processed_frames": float(processed_frames),
            "frames_dropped": float(replay.frames_dropped),
            "detections_per_frame": sum(len(ctx.detections) for ctx in frames) / processed_frames,
            "tracks_per_frame": sum(len(ctx.tracks) for ctx in frames) / processed_frames,
        }

    def compare_with_previous(self, run_id: str) -> dict:
        raise NotImplementedError("Run comparisons are not implemented yet")
