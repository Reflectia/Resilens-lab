from abc import ABC, abstractmethod

from resilens.core.contracts import FrameContext


class Stage(ABC):
    name: str = "stage"

    def setup(self) -> None:  # noqa: B027 - optional hook
        """Load models / resources once before the first frame."""
        pass

    @abstractmethod
    def process(self, ctx: FrameContext) -> FrameContext:
        """Read the frame context, add this stage's result and return it."""
        pass

    def teardown(self) -> None:  # noqa: B027 - optional hook
        """Release resources after the last frame."""
        pass
