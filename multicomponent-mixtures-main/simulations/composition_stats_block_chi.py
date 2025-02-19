#!/usr/bin/env python3 -m modelrunner
"""
This script calculates composition statistics, i.e., how the fractions are distributed
among phases, for interaction matrices with block structure.

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
    num_blocks: int,  # number of blocks in the interaction matrix
    chi_attract: float,  # mean value of the attractive interaction
    chi_repel: float = 0,  # mean value of the repulsive interaction
    chi_noise: float = 0,  # (multiplicative) noise added on top of the block structure
    composition_distribution: str = "simplex_uniform",  # initial comp. distribution
    comp_sigma: float = 1.0,  # breadth of the random composition distribution
    num_phases=None,  # number of initial phases to try
    repetitions: int = 3,  # how many repetitions of different random compositions
    t_range: float = 1e3,
    seed_chi=None,  # seed of the random number generator for choosing chi matrix
    seed_mix=None,  # seed of the random number generator for choosing mixtures
):
    """determine the distribution of the number of phases for block chi matrices"""
    if num_phases is None:
        num_phases = 2 * num_comp
    evolve_params = {
        "t_range": t_range,
        "dt": 1e-1,
        "interval": 1.0,
        "tolerance": 1e-5,
        "progress": False,
        "save_intermediate_data": False,
    }
    # list of enrichment thresholds that are measured
    enriched_fraction = [1.1, 1.2, 1.3, 1.4, 1.5]

    # setup progress bar
    progress_bar = tqdm(total=repetitions, leave=False, mininterval=1)  # 5 * 60)

    # iterate over different realizations of the interaction matrix chis
    diagnostics = {"reached_final_time": 0, "simulation_aborted": 0, "t_final": []}
    start = time.time()

    # choose a random free energy
    rng_chi = None if seed_chi is None else np.random.default_rng(seed_chi)
    f = mm.FloryHuggins.from_block_structure(
        num_comp, num_blocks, chi_attract, chi_repel, noise=chi_noise, rng=rng_chi
    )

    # prepare simulation
    rng_mix = None if seed_mix is None else np.random.default_rng(seed_mix)
    m = mm.MultiphaseSystem.from_random_composition(
        f, num_phases, dist=composition_distribution, sigma=comp_sigma, rng=rng_mix
    )
    r = mm.RelaxationDynamics(m)

    # iterate over different initial conditions
    angles, c_tot = [], []
    enriched = {f: [] for f in enriched_fraction}
    for _ in range(repetitions):
        # choose a new initial composition
        r.mixture.free_energy.set_block_structure(
            num_blocks, chi_attract, chi_repel, noise=chi_noise, rng=rng_chi
        )
        r.mixture.set_random_composition(
            dist=composition_distribution, sigma=comp_sigma, rng=rng_mix
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
            diagnostics["t_final"].append(ts)

            angles.append(result.composition_angles())
            phis = result.get_clusters()
            c_tot.append(phis.sum(axis=1))
            for f in enriched_fraction:
                enriched[f].append(result.count_enriched_components(f))

        # advance progress bar
        progress_bar.update(1)

    # add diagnostic information
    t_final = np.array(diagnostics["t_final"])
    diagnostics["t_final"] = {
        "min": float(t_final.min()),
        "mean": float(t_final.mean()),
        "std": float(t_final.std()),
        "max": float(t_final.max()),
    }
    diagnostics["runtime"] = time.time() - start

    # compile result
    results = {
        "evolve_params": evolve_params,
        "mixing_angles": np.concatenate(angles),
        "total_concentrations": np.concatenate(c_tot),
        "diagnostics": diagnostics,
    }
    for k, v in enriched.items():
        results[f"enriched_{k}"] = np.concatenate(v)

    return results
