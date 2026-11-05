from pathlib import Path

from resilens.config.loader import POLICY_RULES, load_policy_config
from resilens.config.schema import PolicyConfig
from resilens.core.contracts import Action, DegradationType


class RuleTable:
    """Rule table of configs/policy/rules.yaml: (degradation, level, profile) -> base action."""

    def __init__(self, config: PolicyConfig) -> None:
        self.config = config
        self._actions: dict[tuple[DegradationType, int], dict[str, Action]] = {
            (degradation, rule.level): rule.actions for rule in config.rules for degradation in rule.types
        }

    @classmethod
    def load(cls, path: Path = POLICY_RULES) -> "RuleTable":
        return cls(load_policy_config(path))

    def profiles(self) -> set[str]:
        """Profiles that every rule has an action for."""
        return set.intersection(*(set(actions) for actions in self._actions.values())) if self._actions else set()

    def lookup(self, degradation: DegradationType, level: int, profile: str) -> Action:
        """Base action; default_action for unknown, none and degradations or levels without a rule."""
        actions = self._actions.get((degradation, level))
        if actions is None:
            return self.config.default_action
        if profile not in actions:
            raise ValueError(f"Policy rules for {degradation} level {level} have no action for profile '{profile}'")
        return actions[profile]

    def downgrade(self, action: Action) -> Action:
        """One step down along downgrade_order; the last action stays."""
        order = self.config.downgrade_order
        return order[min(order.index(action) + 1, len(order) - 1)]
