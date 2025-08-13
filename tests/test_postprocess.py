import pandas as pd
import numpy as np

from src.models.postprocess import apply_min_duration, confirm_by_probability


def test_apply_min_duration_merges_short_runs():
    # labels: short run of '1' of length 1 between zeros
    idx = pd.date_range("2000-01-01", periods=6, freq="M")
    s = pd.Series([0, 0, 1, 0, 0, 0], index=idx, dtype="Int64")
    out = apply_min_duration(s, k=2)
    assert (out.values == np.array([0, 0, 0, 0, 0, 0])).all()


def test_confirm_by_probability_requires_consecutive():
    idx = pd.date_range("2000-01-01", periods=6, freq="M")
    # two-state probabilities; start state 0 with high prob, then state 1 with high prob for two periods
    probs = pd.DataFrame(
        {
            "state_0": [0.9, 0.9, 0.2, 0.2, 0.2, 0.9],
            "state_1": [0.1, 0.1, 0.8, 0.8, 0.8, 0.1],
        },
        index=idx,
    )
    # Need 2 consecutive periods >= 0.7 to switch
    labels = confirm_by_probability(probs, threshold=0.7, consecutive=2)
    # Should remain in 0 for first two, switch to 1 on the 4th point (after two confirmations)
    expected = pd.Series([0, 0, 0, 1, 1, 1], index=idx, dtype="Int64")
    assert (labels.values == expected.values).all()


