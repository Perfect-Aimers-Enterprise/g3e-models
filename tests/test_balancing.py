from shared.balancing import compute_balancing_plan, expand_samples, summarize_expansion


REAL_COUNTS = {"normal": 12693, "hazard": 2073, "caution": 1543, "potential_threat": 68}


def test_capped_strategy_never_downsamples():
    plan = compute_balancing_plan(REAL_COUNTS, strategy="capped", max_multiplier=20.0)
    for cls, orig in plan.original_counts.items():
        assert plan.target_counts[cls] >= orig


def test_capped_strategy_respects_max_multiplier():
    plan = compute_balancing_plan(REAL_COUNTS, strategy="capped", max_multiplier=20.0)
    assert plan.multipliers["potential_threat"] == 20.0
    assert plan.target_counts["potential_threat"] == 68 * 20


def test_majority_class_is_never_oversampled():
    plan = compute_balancing_plan(REAL_COUNTS, strategy="capped", max_multiplier=20.0)
    assert plan.multipliers["normal"] == 1.0
    assert plan.target_counts["normal"] == REAL_COUNTS["normal"]


def test_match_majority_produces_extreme_multiplier_for_rare_class():
    plan = compute_balancing_plan(REAL_COUNTS, strategy="match_majority")
    assert plan.target_counts["potential_threat"] == REAL_COUNTS["normal"]
    assert plan.multipliers["potential_threat"] > 100


def test_none_strategy_leaves_everything_unchanged():
    plan = compute_balancing_plan(REAL_COUNTS, strategy="none")
    assert plan.target_counts == REAL_COUNTS
    assert all(m == 1.0 for m in plan.multipliers.values())


def test_explicit_multiplier_overrides_strategy_for_named_class_only():
    plan = compute_balancing_plan(
        REAL_COUNTS, strategy="capped", max_multiplier=20.0, explicit_multipliers={"potential_threat": 50}
    )
    assert plan.target_counts["potential_threat"] == 68 * 50
    assert plan.multipliers["hazard"] < 20.0
    assert plan.multipliers["hazard"] > 1.0


def test_empty_counts_returns_empty_plan():
    plan = compute_balancing_plan({})
    assert plan.original_counts == {}
    assert plan.target_counts == {}


def test_unknown_strategy_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_balancing_plan(REAL_COUNTS, strategy="not_a_real_strategy")


def test_expand_samples_round_robins_deterministically():
    samples = [{"id": f"s{i}", "state": "rare"} for i in range(2)]
    plan = compute_balancing_plan({"rare": 2}, strategy="none")
    plan.target_counts["rare"] = 5  # force expansion manually for this test

    expanded = expand_samples(samples, lambda s: s["state"], plan)
    ids = [s["id"] for s, _ in expanded]
    assert ids == ["s0", "s1", "s0", "s1", "s0"]


def test_expand_samples_flags_duplicates_correctly():
    samples = [{"id": "only_one", "state": "rare"}]
    plan = compute_balancing_plan({"rare": 1}, strategy="none")
    plan.target_counts["rare"] = 4

    expanded = expand_samples(samples, lambda s: s["state"], plan)
    flags = [is_dup for _, is_dup in expanded]
    assert flags == [False, True, True, True]


def test_expand_samples_never_expands_below_original_count():
    samples = [{"id": f"s{i}", "state": "common"} for i in range(5)]
    plan = compute_balancing_plan({"common": 5}, strategy="none")

    expanded = expand_samples(samples, lambda s: s["state"], plan)
    assert len(expanded) == 5
    assert all(not is_dup for _, is_dup in expanded)


def test_summarize_expansion_matches_target_counts():
    samples = [{"id": f"s{i}", "state": "a"} for i in range(2)] + [{"id": f"t{i}", "state": "b"} for i in range(3)]
    plan = compute_balancing_plan({"a": 2, "b": 3}, strategy="capped", max_multiplier=3.0)
    expanded = expand_samples(samples, lambda s: s["state"], plan)
    summary = summarize_expansion(expanded, lambda pair: pair["state"])
    assert summary == plan.target_counts
