"""Genetic algorithm for QSAR-guided molecular optimization.

Evolves a population of molecules using BRICS-based crossover and
mutation operators, with fitness scored by a QSAR model or custom function.
"""

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import BRICS

from drugflow.core.constants import (
    GA_CROSSOVER_RATE,
    GA_ELITE_FRACTION,
    GA_MUTATION_RATE,
    GA_NUM_GENERATIONS,
    GA_POPULATION_SIZE,
    GA_TOURNAMENT_SIZE,
)
from drugflow.core.exceptions import GenerationError
from drugflow.core.models import MoleculeDataset, MoleculeRecord, MoleculeStatus
from drugflow.phase3.generation.mutations import random_mutation
from drugflow.utils.chem import mol_to_smiles

logger = logging.getLogger(__name__)


@dataclass
class GeneticAlgorithmConfig:
    """Configuration for the genetic algorithm."""

    population_size: int = GA_POPULATION_SIZE
    n_generations: int = GA_NUM_GENERATIONS
    mutation_rate: float = GA_MUTATION_RATE
    crossover_rate: float = GA_CROSSOVER_RATE
    elite_fraction: float = GA_ELITE_FRACTION
    tournament_size: int = GA_TOURNAMENT_SIZE
    seed: int = 42


def _initialize_population(
    seed_dataset: MoleculeDataset,
    pop_size: int,
    rng: random.Random,
) -> List[Chem.Mol]:
    """Initialize population from seed dataset.

    Samples molecules from the seed dataset with replacement.

    Args:
        seed_dataset: Source molecules.
        pop_size: Target population size.
        rng: Random number generator.

    Returns:
        List of molecules forming the initial population.
    """
    valid = [rec.mol for rec in seed_dataset.valid_records if rec.mol is not None]
    if not valid:
        raise GenerationError("No valid molecules in seed dataset")

    population = []
    for _ in range(pop_size):
        population.append(rng.choice(valid))

    return population


def _evaluate_fitness(
    population: List[Chem.Mol],
    fitness_fn: Callable[[Chem.Mol], float],
) -> np.ndarray:
    """Evaluate fitness of each molecule in the population.

    Args:
        population: List of molecules.
        fitness_fn: Function mapping molecule to fitness score.

    Returns:
        Array of fitness values.
    """
    fitness = np.zeros(len(population))
    for i, mol in enumerate(population):
        try:
            fitness[i] = fitness_fn(mol)
        except Exception:
            fitness[i] = 0.0  # Failed molecules get zero fitness
    return fitness


def _select_parents(
    population: List[Chem.Mol],
    fitness: np.ndarray,
    n_parents: int,
    tournament_size: int = GA_TOURNAMENT_SIZE,
    rng: Optional[random.Random] = None,
) -> List[Chem.Mol]:
    """Select parents via tournament selection.

    Args:
        population: Current population.
        fitness: Fitness array.
        n_parents: Number of parents to select.
        tournament_size: Tournament size.
        rng: Random number generator.

    Returns:
        List of selected parent molecules.
    """
    if rng is None:
        rng = random.Random()

    parents = []
    pop_size = len(population)

    for _ in range(n_parents):
        # Tournament: pick k random individuals, keep the best
        indices = [rng.randint(0, pop_size - 1) for _ in range(tournament_size)]
        best_idx = max(indices, key=lambda i: fitness[i])
        parents.append(population[best_idx])

    return parents


def _crossover_brics(
    parent1: Chem.Mol,
    parent2: Chem.Mol,
    rng: Optional[random.Random] = None,
) -> Optional[Chem.Mol]:
    """BRICS-based crossover between two parent molecules.

    Decomposes both parents into fragments, then reassembles a child
    from mixed fragments.

    Args:
        parent1: First parent molecule.
        parent2: Second parent molecule.
        rng: Random number generator.

    Returns:
        Child molecule or None if crossover fails.
    """
    if rng is None:
        rng = random.Random()

    try:
        frags1 = list(BRICS.BRICSDecompose(parent1))
        frags2 = list(BRICS.BRICSDecompose(parent2))
    except Exception:
        return None

    if not frags1 and not frags2:
        return None

    # Mix fragments from both parents
    all_frags = list(set(frags1 + frags2))
    if not all_frags:
        return None

    # Randomly select a subset of fragments
    n_select = rng.randint(1, min(len(all_frags), 4))
    selected = rng.sample(all_frags, n_select)

    # Convert to mol objects
    frag_mols = []
    for smi in selected:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_mols.append(mol)

    if not frag_mols:
        return None

    # Build a child from fragments
    try:
        builder = BRICS.BRICSBuild(frag_mols)
        children = []
        for mol in builder:
            if mol is not None:
                try:
                    Chem.SanitizeMol(mol)
                    children.append(mol)
                    if len(children) >= 5:
                        break
                except Exception:
                    continue

        if children:
            return rng.choice(children)
    except Exception:
        pass

    return None


def _mutate(
    mol: Chem.Mol,
    mutation_rate: float,
    rng: random.Random,
) -> Chem.Mol:
    """Apply mutation to a molecule with given probability.

    Args:
        mol: Input molecule.
        mutation_rate: Probability of mutation.
        rng: Random number generator.

    Returns:
        Mutated molecule (or original if no mutation applied).
    """
    if rng.random() > mutation_rate:
        return mol

    mutated = random_mutation(mol, rng=rng)
    return mutated if mutated is not None else mol


def run_ga(
    seed_dataset: MoleculeDataset,
    fitness_fn: Callable[[Chem.Mol], float],
    config: Optional[GeneticAlgorithmConfig] = None,
) -> MoleculeDataset:
    """Run the genetic algorithm.

    Args:
        seed_dataset: Initial seed molecules.
        fitness_fn: Fitness function mapping Mol → float.
        config: GA configuration. Uses defaults if None.

    Returns:
        MoleculeDataset of the final population (best molecules).
    """
    if config is None:
        config = GeneticAlgorithmConfig()

    rng = random.Random(config.seed)
    np_rng = np.random.RandomState(config.seed)

    logger.info(
        f"GA: pop={config.population_size}, gen={config.n_generations}, "
        f"mut_rate={config.mutation_rate}, cx_rate={config.crossover_rate}"
    )

    # Initialize
    population = _initialize_population(seed_dataset, config.population_size, rng)
    n_elite = max(1, int(config.population_size * config.elite_fraction))

    best_fitness_history: List[float] = []

    for gen in range(config.n_generations):
        # Evaluate fitness
        fitness = _evaluate_fitness(population, fitness_fn)

        # Track best
        best_idx = np.argmax(fitness)
        best_fitness = fitness[best_idx]
        best_fitness_history.append(float(best_fitness))

        if gen % 10 == 0 or gen == config.n_generations - 1:
            mean_fitness = float(np.mean(fitness))
            logger.info(
                f"  Gen {gen}: best={best_fitness:.4f}, mean={mean_fitness:.4f}"
            )

        # Elitism: keep top-N
        elite_indices = np.argsort(fitness)[-n_elite:]
        elites = [population[i] for i in elite_indices]

        # Generate next generation
        next_gen = list(elites)  # Start with elites

        while len(next_gen) < config.population_size:
            # Select parents
            parents = _select_parents(
                population, fitness, 2,
                tournament_size=config.tournament_size, rng=rng,
            )

            # Crossover
            if rng.random() < config.crossover_rate:
                child = _crossover_brics(parents[0], parents[1], rng=rng)
                if child is None:
                    child = rng.choice(parents)
            else:
                child = rng.choice(parents)

            # Mutation
            child = _mutate(child, config.mutation_rate, rng)
            next_gen.append(child)

        population = next_gen[:config.population_size]

    # Final evaluation
    fitness = _evaluate_fitness(population, fitness_fn)

    # Create output dataset from unique molecules, sorted by fitness
    seen: Set[str] = set()
    records = []
    sorted_indices = np.argsort(fitness)[::-1]

    for idx in sorted_indices:
        mol = population[idx]
        if mol is None:
            continue
        try:
            smi = Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            continue

        if smi in seen:
            continue
        seen.add(smi)

        rec = MoleculeRecord(
            mol=mol,
            source_id=f"ga_{len(records)}",
            smiles=smi,
            status=MoleculeStatus.RAW,
        )
        rec.add_provenance("generated:genetic_algorithm")
        rec.metadata["generation_method"] = "genetic_algorithm"
        rec.metadata["ga_fitness"] = float(fitness[idx])
        rec.properties["ga_fitness"] = float(fitness[idx])
        records.append(rec)

    result = MoleculeDataset(
        records=records,
        name=f"ga_generated_{len(records)}",
    )
    logger.info(
        f"GA complete: {len(records)} unique molecules, "
        f"best fitness={best_fitness_history[-1]:.4f}"
    )
    return result


def qsar_guided_ga(
    seed_dataset: MoleculeDataset,
    model: Any,
    feature_source: str = "descriptors",
    fp_type: Optional[str] = None,
    config: Optional[GeneticAlgorithmConfig] = None,
) -> MoleculeDataset:
    """Run GA with a QSAR model as the fitness function.

    Uses the Phase 2 QSAR model's predict_single() to score molecules
    during the evolutionary process.

    Args:
        seed_dataset: Initial seed molecules.
        model: Trained QSARModel from Phase 2.
        feature_source: Feature source for prediction ("descriptors" or "fingerprints").
        fp_type: Fingerprint type if using fingerprints.
        config: GA configuration.

    Returns:
        MoleculeDataset of evolved molecules.
    """
    from drugflow.phase1.analysis.descriptors import compute_descriptors
    from drugflow.phase1.analysis.fingerprints import compute_morgan
    from drugflow.phase2.qsar.prediction import predict_single

    def qsar_fitness(mol: Chem.Mol) -> float:
        """Fitness function using QSAR model prediction."""
        try:
            # Create a temporary record with computed features
            rec = MoleculeRecord(
                mol=mol,
                source_id="temp",
                smiles=Chem.MolToSmiles(mol, canonical=True),
            )
            # Compute required features
            if feature_source == "descriptors":
                desc = compute_descriptors(mol)
                rec.descriptors = desc
            elif feature_source == "fingerprints":
                fp = compute_morgan(mol)
                fp_key = fp_type or "morgan_r2_2048"
                rec.fingerprints[fp_key] = fp

            return predict_single(model, rec, feature_source=feature_source, fp_type=fp_type)
        except Exception:
            return 0.0

    return run_ga(seed_dataset, qsar_fitness, config=config)
