"""Registry that maps names from YAML configs to stage classes.

Adding a new model = new Stage subclass + @register("name") + a line in a config.
"""

from collections.abc import Callable

from resilens.core.stage import Stage


class StageRegistry:
    def __init__(self) -> None:
        self._stages: dict[str, type[Stage]] = {}

    def register(self, name: str) -> Callable[[type[Stage]], type[Stage]]:
        """Decorator that registers a stage class under a config name."""

        def decorator(stage_class: type[Stage]) -> type[Stage]:
            if name in self._stages and self._stages[name] is not stage_class:
                raise ValueError(f"Stage name '{name}' is already registered by {self._stages[name].__name__}")
            self._stages[name] = stage_class
            return stage_class

        return decorator

    def get(self, name: str) -> type[Stage]:
        if name not in self._stages:
            raise ValueError(f"Unknown stage '{name}'. Available: {', '.join(self.names())}")
        return self._stages[name]

    def create(self, name: str, /, **params) -> Stage:
        """Instantiate a registered stage with parameters from the config."""
        return self.get(name)(**params)

    def names(self) -> list[str]:
        return sorted(self._stages)


registry = StageRegistry()
