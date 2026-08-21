#!/usr/bin/env python3
"""Infer the equilibrium distribution in the six-state, zero-entropy model.

The thermodynamically allowed region is the subset of the probability simplex
that satisfies the entropy-production inequalities for the cooled
distributions. Uniform simplex points are generated with a scrambled Sobol
sequence; every retained point satisfies the inequalities directly.
"""

from pathlib import Path
import pickle

import numpy as np
from scipy.stats import qmc


# Input and output configuration
HERE = Path(__file__).resolve().parent
INPUT_FILE = HERE / "particle_assignments.pkl"
RESULTS_DIR = HERE / "Inference_Results"
OUTPUT_FILE = RESULTS_DIR / "figure_data.npz"
STATE_SUMMARY_FILE = RESULTS_DIR / "state_probability_summary.csv"
SINGLE_TAU_FILE = RESULTS_DIR / "single_condition_recovery.csv"

# Sampling configuration
N_REPLICATES = 16
DRAWS_PER_REPLICATE = 2**24
CHUNK_SIZE = 2**20
RANDOM_SEED = 20_260_814
TOLERANCE = 1e-12

# Model configuration
T_HOT = 500.0
T_COLD = 100.0
BETA = T_HOT / T_COLD
TRUE_EQUILIBRIUM = np.array(
    [0.1154, 0.1807, 0.1383, 0.1271, 0.2776, 0.1609]
)


def load_cooled_distributions():
    """Return cooling times and exact state fractions from the simulations."""
    with INPUT_FILE.open("rb") as stream:
        assignments = pickle.load(stream)

    times = np.array(sorted(assignments), dtype=int)
    cooled = []
    for time in times:
        states = np.asarray(assignments[time])[:, 0].astype(int)
        counts = np.bincount(states - 1, minlength=6)
        cooled.append(counts / counts.sum())
    return times, np.asarray(cooled)


def uniform_simplex_points(engine, size):
    """Generate points uniformly in the six-state probability simplex."""
    cuts = engine.random(size)
    cuts.sort(axis=1)
    points = np.empty((size, 6))
    points[:, 0] = cuts[:, 0]
    points[:, 1:5] = np.diff(cuts, axis=1)
    points[:, 5] = 1.0 - cuts[:, 4]
    return points


def cold_equilibrium_distribution(probabilities):
    """Return the cold equilibrium distribution from manuscript Eq. (8)."""
    log_p = np.log(np.maximum(probabilities, np.finfo(float).tiny))

    # Eq. (8), with s_i = 0 and energies inferred from the hot equilibrium:
    # e_i / (k_B T_h) = -log(p_i), up to an irrelevant additive constant.
    energies = -log_p
    log_cold_weights = -BETA * energies
    log_cold_weights -= log_cold_weights.max(axis=1, keepdims=True)
    cold = np.exp(log_cold_weights)
    return cold / cold.sum(axis=1, keepdims=True)


def system_entropy(probabilities):
    """Return S/k_B from manuscript Eq. (9), with all s_i = 0."""
    log_p = np.log(np.maximum(probabilities, np.finfo(float).tiny))
    return -np.sum(probabilities * log_p, axis=1)


def entropy_productions(probabilities, cooled):
    """Evaluate manuscript Eqs. (11) and (12) without simplification."""
    log_p = np.log(np.maximum(probabilities, np.finfo(float).tiny))

    # From manuscript Eq. (8), with dimensionless microstate entropies s_i=0.
    # Energies are expressed in units of k_B T_h.
    energies = -log_p
    cold = cold_equilibrium_distribution(probabilities)

    hot_energy = np.sum(probabilities * energies, axis=1)
    cold_energy = np.sum(cold * energies, axis=1)
    cooled_energy = energies @ cooled.T

    hot_entropy = system_entropy(probabilities)
    cold_entropy = system_entropy(cold)
    cooled_entropy = system_entropy(cooled)

    # Manuscript Eq. (12): upper bound on cooling entropy production.
    cooling = (
        cooled_entropy[None, :]
        - hot_entropy[:, None]
        - BETA * (cooled_energy - hot_energy[:, None])
    )

    # Manuscript Eq. (11): entropy production during auxiliary relaxation.
    auxiliary = (
        cold_entropy[:, None]
        - cooled_entropy[None, :]
        - BETA * (cold_energy[:, None] - cooled_energy)
    )
    return cooling, auxiliary


def kl_divergence(probabilities, reference):
    return np.sum(probabilities * np.log(probabilities / reference), axis=-1)


def main():
    times, cooled = load_cooled_distributions()
    allowed_chunks = []
    single_count = np.zeros(len(times), dtype=int)
    single_sum = np.zeros_like(cooled)
    single_sum_squares = np.zeros_like(cooled)

    for replicate in range(N_REPLICATES):
        engine = qmc.Sobol(d=5, scramble=True, seed=RANDOM_SEED + replicate)
        accepted_this_replicate = 0

        for _ in range(DRAWS_PER_REPLICATE // CHUNK_SIZE):
            probabilities = uniform_simplex_points(engine, CHUNK_SIZE)
            cooling, auxiliary = entropy_productions(probabilities, cooled)

            # Manuscript Eq. (2): both entropy productions are nonnegative.
            # Manuscript Eq. (3): cooling entropy production increases and
            # auxiliary entropy production decreases with cooling time.
            allowed = probabilities[
                np.all(cooling >= -TOLERANCE, axis=1)
                & np.all(auxiliary >= -TOLERANCE, axis=1)
                & np.all(np.diff(cooling, axis=1) >= -TOLERANCE, axis=1)
                & np.all(np.diff(auxiliary, axis=1) <= TOLERANCE, axis=1)
            ]
            allowed_chunks.append(allowed)
            accepted_this_replicate += len(allowed)

            # With one cooling time, manuscript Eq. (2) supplies both tests.
            for condition in range(len(times)):
                single_allowed = probabilities[
                    (cooling[:, condition] >= -TOLERANCE)
                    & (auxiliary[:, condition] >= -TOLERANCE)
                ]
                single_count[condition] += len(single_allowed)
                single_sum[condition] += single_allowed.sum(axis=0)
                single_sum_squares[condition] += (single_allowed**2).sum(axis=0)

        print(
            f"Replicate {replicate + 1:2d}/{N_REPLICATES}: "
            f"{accepted_this_replicate:,} allowed points"
        )

    allowed = np.vstack(allowed_chunks)
    allowed_mean = allowed.mean(axis=0)
    allowed_std = allowed.std(axis=0)
    single_mean = single_sum / single_count[:, None]
    single_std = np.sqrt(
        np.maximum(single_sum_squares / single_count[:, None] - single_mean**2, 0)
    )

    # Thermodynamic curves used in Figure S12.
    cold_equilibrium = cold_equilibrium_distribution(TRUE_EQUILIBRIUM[None, :])[0]
    cooling_entropy_production, auxiliary_entropy_production = (
        entropy_productions(TRUE_EQUILIBRIUM[None, :], cooled)
    )
    cooling_entropy_production = cooling_entropy_production[0]
    auxiliary_entropy_production = auxiliary_entropy_production[0]
    cooling_energy_change = np.sum(
        (cooled - TRUE_EQUILIBRIUM) * (-np.log(TRUE_EQUILIBRIUM)), axis=1
    )
    cooling_entropy_change = (
        -np.sum(cooled * np.log(cooled), axis=1)
        + np.sum(TRUE_EQUILIBRIUM * np.log(TRUE_EQUILIBRIUM))
    )
    cooled_kl = kl_divergence(cooled, TRUE_EQUILIBRIUM)
    single_kl = kl_divergence(single_mean, TRUE_EQUILIBRIUM)

    # Independent final check of every saved point.
    cooling, auxiliary = entropy_productions(allowed, cooled)
    assert np.allclose(allowed.sum(axis=1), 1.0)
    assert cooling.min() >= -TOLERANCE
    assert auxiliary.min() >= -TOLERANCE
    assert np.diff(cooling, axis=1).min() >= -TOLERANCE
    assert np.diff(auxiliary, axis=1).max() <= TOLERANCE

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT_FILE,
        cooling_times=times,
        cooled_probabilities=cooled,
        true_equilibrium=TRUE_EQUILIBRIUM,
        cold_equilibrium=cold_equilibrium,
        allowed_probabilities=allowed,
        allowed_mean=allowed_mean,
        allowed_std=allowed_std,
        single_condition_mean=single_mean,
        single_condition_std=single_std,
        cooled_kl_to_equilibrium=cooled_kl,
        single_condition_kl_to_equilibrium=single_kl,
        cooling_entropy_production=cooling_entropy_production,
        auxiliary_entropy_production=auxiliary_entropy_production,
        cooling_energy_change=cooling_energy_change,
        cooling_entropy_change=cooling_entropy_change,
        total_simplex_draws=N_REPLICATES * DRAWS_PER_REPLICATE,
        allowed_count=len(allowed),
    )

    np.savetxt(
        STATE_SUMMARY_FILE,
        np.column_stack(
            [
                np.arange(1, 7),
                TRUE_EQUILIBRIUM,
                cooled[0],
                allowed_mean,
                allowed_std,
            ]
        ),
        delimiter=",",
        header="state,true_equilibrium,fastest_cooled,allowed_mean,allowed_std",
        comments="",
    )
    np.savetxt(
        SINGLE_TAU_FILE,
        np.column_stack([times, cooled_kl, single_kl, single_count]),
        delimiter=",",
        header="cooling_time,cooled_kl,inferred_kl,allowed_count",
        comments="",
    )

    print(f"\nSaved {len(allowed):,} allowed points to {OUTPUT_FILE.name}")
    print("Allowed-region mean:", allowed_mean)
    print("Allowed-region std: ", allowed_std)
    print(
        "KL(mean || true equilibrium):",
        kl_divergence(allowed_mean, TRUE_EQUILIBRIUM),
    )


if __name__ == "__main__":
    main()
