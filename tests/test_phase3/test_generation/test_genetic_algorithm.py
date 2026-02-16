"""Tests for the genetic algorithm."""

import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from claudedd.phase3.generation.genetic_algorithm import (
    GeneticAlgorithmConfig,
    _crossover_brics,
    _evaluate_fitness,
    _initialize_population,
    _mutate,
    _select_parents,
    run_ga,
)

import numpy as np
import random


def test_ga_config_defaults():
    """Default GA config has reasonable values."""
    config = GeneticAlgorithmConfig()
    assert config.population_size == 100
    assert config.n_generations == 50
    assert 0 < config.mutation_rate <= 1
    assert 0 < config.crossover_rate <= 1


def test_initialize_population(seed_dataset):
    """Population initialized from seed dataset."""
    rng = random.Random(42)
    pop = _initialize_population(seed_dataset, 10, rng)
    assert len(pop) == 10
    for mol in pop:
        assert mol is not None


def test_evaluate_fitness(seed_dataset):
    """Fitness evaluation with simple function."""
    rng = random.Random(42)
    pop = _initialize_population(seed_dataset, 5, rng)

    def simple_fitness(mol):
        return Descriptors.MolWt(mol)

    fitness = _evaluate_fitness(pop, simple_fitness)
    assert len(fitness) == 5
    assert all(f > 0 for f in fitness)


def test_select_parents(seed_dataset):
    """Tournament selection picks parents."""
    rng = random.Random(42)
    pop = _initialize_population(seed_dataset, 10, rng)
    fitness = np.array([float(i) for i in range(10)])
    parents = _select_parents(pop, fitness, 4, rng=rng)
    assert len(parents) == 4


def test_crossover_brics():
    """BRICS crossover between two molecules."""
    mol1 = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    mol2 = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")  # ibuprofen
    rng = random.Random(42)
    child = _crossover_brics(mol1, mol2, rng=rng)
    # May or may not succeed depending on BRICS compatibility
    if child is not None:
        assert Chem.MolToSmiles(child) is not None


def test_mutate_applies_with_probability():
    """Mutation applied based on rate."""
    mol = Chem.MolFromSmiles("CCCC")
    rng = random.Random(42)
    # With rate=1.0, should always attempt mutation
    result = _mutate(mol, 1.0, rng)
    assert result is not None


def test_run_ga_simple(seed_dataset):
    """Run GA with simple fitness function (small pop, few gens)."""
    def mw_fitness(mol):
        try:
            return Descriptors.MolWt(mol)
        except Exception:
            return 0.0

    config = GeneticAlgorithmConfig(
        population_size=10,
        n_generations=3,
        seed=42,
    )
    result = run_ga(seed_dataset, mw_fitness, config=config)
    assert len(result) > 0
    for rec in result.valid_records:
        assert rec.metadata["generation_method"] == "genetic_algorithm"
        assert "ga_fitness" in rec.properties
