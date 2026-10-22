"""Registry that maps names from YAML configs to stage classes.

Adding a new model = new Stage subclass + @register("name") + a line in a config.
"""

from collections.abc import Callable

from resilens.core.stage import Stage


class StageRegistry:
    def register(self, name: str) -> Callable[[type[Stage]], type[Stage]]:
        """Decorator that registers a stage class under a config name."""
        pass

    def create(self, name: str, **params) -> Stage:
        """Instantiate a registered stage with parameters from the config."""
        pass

    def names(self) -> list[str]:
        pass


registry = StageRegistry()
