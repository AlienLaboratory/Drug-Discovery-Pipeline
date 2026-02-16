"""Tests for acquisition functions."""

import pytest

from claudedd.phase3.active_learning.acquisition import (
    balanced_acquisition,
    diversity_acquisition,
    greedy_acquisition,
    ucb_acquisition,
    uncertainty_acquisition,
)


def test_greedy_acquisition(trained_rf_model, computed_dataset):
    """Greedy acquisition selects top predicted."""
    selected = greedy_acquisition(
        computed_dataset, trained_rf_model, batch_size=3,
    )
    assert len(selected) <= 3
    assert len(selected) > 0


def test_uncertainty_acquisition(trained_rf_model, computed_dataset):
    """Uncertainty acquisition selects high-uncertainty molecules."""
    selected = uncertainty_acquisition(
        computed_dataset, trained_rf_model, batch_size=3,
    )
    assert len(selected) <= 3
    assert len(selected) > 0


def test_ucb_acquisition(trained_rf_model, computed_dataset):
    """UCB acquisition balances prediction and uncertainty."""
    selected = ucb_acquisition(
        computed_dataset, trained_rf_model, batch_size=3, kappa=1.0,
    )
    assert len(selected) <= 3
    assert len(selected) > 0


def test_diversity_acquisition(computed_dataset):
    """Diversity acquisition selects diverse molecules."""
    selected = diversity_acquisition(
        computed_dataset, batch_size=3, fp_type="morgan_r2_2048",
    )
    assert len(selected) <= 3
    assert len(selected) > 0


def test_balanced_acquisition(trained_rf_model, computed_dataset):
    """Balanced acquisition combines strategies."""
    selected = balanced_acquisition(
        computed_dataset, trained_rf_model,
        batch_size=3, exploration_weight=0.5,
    )
    assert len(selected) <= 3
    assert len(selected) > 0
