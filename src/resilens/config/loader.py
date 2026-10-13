from pathlib import Path

from resilens.config.schema import ModeConfig, ProfileConfig, RunConfig


def load_mode(name_or_path: str | Path) -> ModeConfig:
    """Load a mode preset from configs/modes/."""
    pass


def load_profile(name_or_path: str | Path) -> ProfileConfig:
    """Load a resource profile from configs/profiles/."""
    pass


def build_run_config(mode: str, profile: str, source: Path, **overrides) -> RunConfig:
    """Combine mode, profile and run options into a validated RunConfig."""
    pass
