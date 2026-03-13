import math
import numpy as np
import pandas as pd
import pytest
from netball_model.model.train import NetballModel, NON_FEATURE_COLUMNS


def test_sample_weight_in_non_feature_columns():
    assert "_sample_weight" in NON_FEATURE_COLUMNS


def test_weight_calculation():
    """Verify exponential decay formula."""
    lambda_ = 0.5
    assert abs(math.exp(-lambda_ * 0) - 1.0) < 0.01
    assert abs(math.exp(-lambda_ * 1) - 0.607) < 0.01
    assert abs(math.exp(-lambda_ * 2) - 0.368) < 0.01


def test_train_with_sample_weights(dummy_feature_df):
    """Model should accept _sample_weight column without error."""
    df = dummy_feature_df(100)
    df["_sample_weight"] = np.random.default_rng(42).uniform(0.1, 1.0, 100)

    model = NetballModel()
    model.train(df)

    assert "_sample_weight" not in model.feature_columns
    preds = model.predict(df.drop(columns=["_sample_weight"]))
    assert len(preds) == 100


def test_train_without_sample_weights_still_works(dummy_feature_df):
    """Backward compatibility: train without _sample_weight."""
    df = dummy_feature_df(100)
    model = NetballModel()
    model.train(df)
    assert len(model.feature_columns) > 0
