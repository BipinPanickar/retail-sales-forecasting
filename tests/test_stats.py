import numpy as np

from src.evaluation.significance import dm_test, holm


def test_dm_detects_clearly_better_model():
    rng = np.random.RandomState(0)
    y = rng.normal(100, 5, 80)
    good, bad = y + rng.normal(0, 1, 80), y + rng.normal(0, 6, 80)
    r = dm_test(y, good, bad, h=1)
    assert r["mean_loss_diff"] < 0 and r["p_value"] < 0.01


def test_dm_identical_error_magnitudes_give_zero_difference():
    # b's errors are a's errors in reverse order: same total absolute loss -> no difference
    rng = np.random.RandomState(1)
    y = rng.normal(100, 5, 80)
    err = rng.normal(0, 3, 80)
    r = dm_test(y, y + err, y + err[::-1], h=1)
    assert abs(r["mean_loss_diff"]) < 1e-9 and r["p_value"] > 0.99


def test_holm_is_monotone_and_bounded():
    adj = holm([0.01, 0.04, 0.03])
    assert (adj >= np.array([0.01, 0.04, 0.03])).all() and adj.max() <= 1.0
