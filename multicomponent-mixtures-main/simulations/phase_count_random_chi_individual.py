#!/usr/bin/env python3 -m modelrunner
"""
This script determines the phase count distribution for random interaction matrices.
Here, the ensemble average over initial conditions is performed separately for each
interaction matrix. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

import time

import numpy as np
from tqdm.auto import tqdm

from modelrunner import make_model

import multicomp as mm


@make_model
def run(
    num_comp: int,  # number of components
    chi_mean: float,  # mean value of the interaction
    chi_std: float = 0,  # standard deviation of the interaction
    comp_dist: str = "lognormal",  # distribution of the initial composition
    comp_sigma: float = 1.0,  # breadth of the random composition distribution
    num_phases=None,  # number of initial phases to try
    chi_repetitions: int = 2,  # how many random compositions are tested
    init_repetitions: int = 3,  # how many initial conditions are tested
    t_range: float = 1e3,
    ensemble_id: int = 0,  # id to distinguish different runs of the same code
    seed_chi=None,  # seed of the random number generator for choosing chi matrix
    seed_mix=None,  # seed of the random number generator for choosing mixtures
):
    """determine the distribution of the number of phases for random chi matrices"""
    if num_phases is None:
        num_phases = num_comp + 2
    evolve_params = {
        "t_range": t_range,
        "dt": 1e-1,
        "interval": 1.0,
        "tolerance": 1e-5,
        "progress": False,
        "save_intermediate_data": False,
    }

    # iterate over different realizations of the interaction matrix chis
    diagnostics = {"reached_final_time": 0, "simulation_aborted": 0, "t_final": []}
    start = time.time()

    # choose a random free energy
    rng_chi = None if seed_chi is None else np.random.default_rng(seed_chi)
    f = mm.FloryHuggins.from_random_normal(num_comp, chi_mean, chi_std, rng=rng_chi)

    # prepare simulation
    rng_mix = None if seed_mix is None else np.random.default_rng(seed_mix)
    m = mm.MultiphaseSystem.from_random_composition(
        f, num_phases, dist=comp_dist, sigma=comp_sigma, rng=rng_mix
    )
    r = mm.RelaxationDynamics(m)

    # iterate over different initial conditions
    cluster_counts, chi_matrices = [], []
    for _ in tqdm(range(chi_repetitions), leave=False):
        # choose a new initial composition
        r.mixture.free_energy.set_random_chis(chi_mean, chi_std, rng=rng_chi)

        counts = []
        for _ in range(init_repetitions):
            r.mixture.set_random_composition(
                dist=comp_dist, sigma=comp_sigma, rng=rng_mix
            )

            # run simulation
            try:
                ts, result = r.evolve(**evolve_params)
            except RuntimeError:
                # simulation could not finish
                diagnostics["simulation_aborted"] += 1
            else:
                # finalize
                if np.isclose(ts, t_range):
                    diagnostics["reached_final_time"] += 1
                counts.append(result.count_clusters())
                diagnostics["t_final"].append(ts)

        # store result
        cluster_counts.append(np.bincount(counts).tolist())
        chi_matrices.append(r.mixture.free_energy.chis)

    # add diagnostic information
    t_final = np.array(diagnostics["t_final"])
    diagnostics["t_final"] = {
        "min": float(t_final.min()),
        "mean": float(t_final.mean()),
        "std": float(t_final.std()),
        "max": float(t_final.max()),
        "ensemble_id": ensemble_id,
    }
    diagnostics["runtime"] = time.time() - start

    # compile result
    return {
        "evolve_params": evolve_params,
        "cluster_counts": cluster_counts,
        "chi_matrices": np.array(chi_matrices),
        "diagnostics": diagnostics,
    }
