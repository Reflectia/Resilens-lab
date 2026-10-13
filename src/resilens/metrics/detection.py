"""Detection quality via pycocotools (annotated videos only)"""

from pathlib import Path

from resilens.core.contracts import FrameContext


def compute_detection_metrics(frames: list[FrameContext], annotations: Path) -> dict[str, float]:
    pass
