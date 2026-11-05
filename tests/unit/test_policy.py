from pathlib import Path

import pytest
from pydantic import ValidationError

from resilens.config.loader import load_profile
from resilens.config.schema import PolicyConfig, ProfileConfig
from resilens.core.contracts import Action, BudgetState, DegradationEstimate, DegradationType
from resilens.policy.engine import PolicyEngine
from resilens.policy.rules import RuleTable
from resilens.testing.mocks import mock_frames

FOG, RAIN = DegradationType.FOG, DegradationType.RAIN
LOW_LIGHT, BLUR = DegradationType.LOW_LIGHT, DegradationType.MOTION_BLUR
ENHANCE, SKIP, BYPASS, DOWNSCALE = Action.ENHANCE, Action.SKIP, Action.BYPASS, Action.DOWNSCALE
NO_BUDGET = BudgetState()

# Task 7, rule table version 0.1: (type, level) -> actions for minimal, standard, performance
SPEC_TABLE = {
    (RAIN, 1): (BYPASS, SKIP, ENHANCE),
    (FOG, 1): (BYPASS, SKIP, ENHANCE),
    (RAIN, 2): (SKIP, ENHANCE, ENHANCE),
    (FOG, 2): (SKIP, ENHANCE, ENHANCE),
    (LOW_LIGHT, 1): (BYPASS, BYPASS, BYPASS),
    (LOW_LIGHT, 2): (BYPASS, BYPASS, BYPASS),
    (BLUR, 1): (BYPASS, BYPASS, BYPASS),
    (BLUR, 2): (BYPASS, BYPASS, BYPASS),
    (DegradationType.UNKNOWN, 0): (BYPASS, BYPASS, BYPASS),
    (DegradationType.NONE, 0): (BYPASS, BYPASS, BYPASS),
}
PROFILES = ("minimal", "standard", "performance")


@pytest.fixture
def rules(configs_dir: Path) -> RuleTable:
    return RuleTable.load(configs_dir / "policy" / "rules.yaml")


def estimate(degradation: DegradationType, level: int) -> DegradationEstimate:
    return DegradationEstimate(degradation, level, confidence=1.0)


def engine(rules: RuleTable, profile: str = "standard", **limits) -> PolicyEngine:
    return PolicyEngine(rules, ProfileConfig(name=profile, **limits))


def config_with(**changes) -> dict:
    data = {"rules": [{"types": ["fog"], "level": 1, "actions": {"standard": "enhance"}}]}
    return {**data, **changes}


@pytest.mark.parametrize(("degradation", "level"), list(SPEC_TABLE))
@pytest.mark.parametrize("profile", PROFILES)
def test_rule_table_matches_spec(rules: RuleTable, degradation, level: int, profile: str) -> None:
    assert rules.lookup(degradation, level, profile) == SPEC_TABLE[degradation, level][PROFILES.index(profile)]


def test_profile_presets_are_covered_by_rules(rules: RuleTable) -> None:
    assert rules.profiles() == set(PROFILES)


def test_level_without_rule_gets_default_action(rules: RuleTable) -> None:
    assert rules.lookup(FOG, 0, "standard") == BYPASS


def test_profile_without_action_is_an_error(rules: RuleTable) -> None:
    with pytest.raises(ValueError, match="no action for profile 'tiny'"):
        rules.lookup(FOG, 1, "tiny")
    with pytest.raises(ValueError, match="no actions for profile 'tiny'"):
        engine(rules, "tiny")


def test_downgrade_order_from_spec(rules: RuleTable) -> None:
    assert [rules.downgrade(a) for a in (ENHANCE, SKIP, BYPASS, DOWNSCALE)] == [SKIP, BYPASS, DOWNSCALE, DOWNSCALE]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"rules": [{"types": ["fog"], "level": 1, "actions": {"standard": "downscale"}}]}, "not allowed in rules"),
        ({"rules": [{"types": ["unknown"], "level": 1, "actions": {"standard": "skip"}}]}, "cannot have rules"),
        ({"rules": [{"types": ["fog"], "level": 3, "actions": {"standard": "skip"}}]}, "less than or equal to 2"),
        ({"rules": [{"types": ["smoke"], "level": 1, "actions": {"standard": "skip"}}]}, "types.0"),
        (
            {
                "rules": [
                    {"types": ["fog", "rain"], "level": 1, "actions": {}},
                    {"types": ["fog"], "level": 1, "actions": {}},
                ]
            },
            "more than one rule for fog level 1",
        ),
        ({"downgrade_order": ["enhance", "bypass", "downscale"]}, "each of"),
        ({"default_action": "downscale"}, "default_action"),
        ({"skip_every_n": 0}, "skip_every_n"),
        ({"budget": {"latency_metric": "max"}}, "latency_metric"),
    ],
    ids=["downscale_rule", "unknown_rule", "level_3", "unknown_type", "duplicate", "order", "default", "n", "metric"],
)
def test_invalid_rules_are_rejected(changes: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PolicyConfig(**config_with(**changes))


@pytest.mark.parametrize("profile", PROFILES)
def test_policy_accepts_profile_presets(rules: RuleTable, profile: str) -> None:
    assert PolicyEngine(rules, load_profile(profile)).profile.name == profile


def test_decision_without_budget_problems(rules: RuleTable) -> None:
    decision = engine(rules).decide(estimate(FOG, 2), NO_BUDGET)

    assert (decision.action, decision.base_action, decision.downgraded) == (ENHANCE, ENHANCE, False)
    assert decision.enhance and decision.resize_to is None
    assert decision.reason == "fog level 2 on standard: enhance"


def test_missing_estimate_counts_as_unknown(rules: RuleTable) -> None:
    decision = engine(rules).decide(None, NO_BUDGET)

    assert decision.action == BYPASS
    assert decision.reason.startswith("unknown level 0")


def test_skip_enhances_every_nth_frame(rules: RuleTable) -> None:
    policy = engine(rules)  # rain level 1 on standard -> skip, N = 3

    decisions = [policy.decide(estimate(RAIN, 1), NO_BUDGET) for _ in range(7)]

    assert {d.action for d in decisions} == {SKIP}
    assert [d.enhance for d in decisions] == [True, False, False, True, False, False, True]


def test_skip_counter_restarts_with_setup(rules: RuleTable) -> None:
    policy = engine(rules)
    policy.decide(estimate(RAIN, 1), NO_BUDGET)

    policy.setup()

    assert policy.decide(estimate(RAIN, 1), NO_BUDGET).enhance


@pytest.mark.parametrize(
    ("degradation", "level", "expected"),
    [(FOG, 2, SKIP), (RAIN, 1, BYPASS), (LOW_LIGHT, 1, DOWNSCALE), (DegradationType.UNKNOWN, 0, DOWNSCALE)],
)
def test_latency_over_budget_lowers_action_one_step(rules: RuleTable, degradation, level: int, expected) -> None:
    policy = engine(rules, frame_budget_ms=40, downscale_resolution=(960, 540))

    decision = policy.decide(estimate(degradation, level), BudgetState(latency_ms=55))

    assert (decision.action, decision.downgraded) == (expected, True)
    assert "latency 55 ms > budget 40 ms" in decision.reason


def test_downscale_carries_profile_resolution(rules: RuleTable) -> None:
    policy = engine(rules, frame_budget_ms=40, downscale_resolution=(960, 540))

    decision = policy.decide(estimate(LOW_LIGHT, 2), BudgetState(latency_ms=55))

    assert (decision.action, decision.resize_to, decision.enhance) == (DOWNSCALE, (960, 540), False)


def test_downscale_without_profile_resolution_keeps_frame(rules: RuleTable) -> None:
    decision = engine(rules, frame_budget_ms=40).decide(estimate(LOW_LIGHT, 2), BudgetState(latency_ms=55))

    assert (decision.action, decision.resize_to) == (DOWNSCALE, None)
    assert "no downscale_resolution" in decision.reason


def test_latency_within_budget_keeps_action(rules: RuleTable) -> None:
    decision = engine(rules, frame_budget_ms=40).decide(estimate(FOG, 2), BudgetState(latency_ms=40))

    assert (decision.action, decision.downgraded) == (ENHANCE, False)


@pytest.mark.parametrize(
    ("budget", "limits", "message"),
    [
        (BudgetState(ram_mb=920), {"memory_limit_mb": 1000}, "RAM 920 MB >= 90% of limit 1000 MB"),
        (BudgetState(cpu_percent=190), {"cpu_limit_percent": 200}, "CPU 190 % >= 90% of limit 200 %"),
        (BudgetState(gpu_memory_mb=1900), {"gpu_memory_limit_mb": 2000}, "GPU memory 1900 MB >= 90%"),
    ],
    ids=["ram", "cpu", "gpu"],
)
def test_resources_near_limit_lower_action(rules: RuleTable, budget: BudgetState, limits: dict, message: str) -> None:
    decision = engine(rules, **limits).decide(estimate(FOG, 2), budget)

    assert decision.action == SKIP
    assert message in decision.reason


def test_resources_below_limit_keep_action(rules: RuleTable) -> None:
    budget = BudgetState(ram_mb=800, cpu_percent=100, gpu_memory_mb=100)
    policy = engine(rules, memory_limit_mb=1000, cpu_limit_percent=200, gpu_memory_limit_mb=2000)

    assert policy.decide(estimate(FOG, 2), budget).action == ENHANCE


def test_unset_limits_are_not_checked(rules: RuleTable) -> None:
    budget = BudgetState(latency_ms=10_000, ram_mb=1e6, cpu_percent=1e4, gpu_memory_mb=1e6)

    assert engine(rules).decide(estimate(FOG, 2), budget).action == ENHANCE


def test_several_problems_still_lower_one_step(rules: RuleTable) -> None:
    budget = BudgetState(latency_ms=100, ram_mb=990)

    decision = engine(rules, frame_budget_ms=40, memory_limit_mb=1000).decide(estimate(FOG, 2), budget)

    assert decision.action == SKIP
    assert "latency" in decision.reason and "RAM" in decision.reason


def test_stage_reads_estimate_and_budget(rules: RuleTable) -> None:
    budget = BudgetState(latency_ms=100)
    policy = PolicyEngine(rules, ProfileConfig(name="standard", frame_budget_ms=40), budget_source=lambda: budget)
    ctx = next(mock_frames(1))
    ctx.degradation = estimate(FOG, 2)

    decision = policy.process(ctx).decision

    assert (decision.action, decision.budget) == (SKIP, budget)
