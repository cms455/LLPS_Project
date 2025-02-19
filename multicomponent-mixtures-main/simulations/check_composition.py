#!/usr/bin/env python3 -m modelrunner
"""
This script takes optimized matrices (read in from the results of an optimization run)
and calculates composition statistics, i.e., how the fractions in optimized systems are
distributed.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

import numpy as np
from scipy import cluster
from tqdm.auto import tqdm

from modelrunner import Result, make_model

import multicomp as mm


@make_model
def run(
    input_path: str,  #  path to the result file from which the data is read
):
    """determine the composition statistics of states

    Args:
        result_path (str):
            Path to result file where information about individual is stored
    """
    # list of enrichment thresholds that are measured
    enriched_fraction = [1.1, 1.2, 1.3, 1.4, 1.5]

    # read result
    result = Result.from_file(input_path)
    try:
        chis = np.array(list(result.result["final_chis"].values()))
    except TypeError:
        print(result.result)
        raise

    # extract parameters for the individual and the population
    params_ind = {
        k: v.default_value for k, v in mm.Individual.get_parameters(sort=False).items()
    }
    for key, value in result.parameters.items():
        if key in params_ind:
            params_ind[key] = value

    # construct the population from the data in the input
    individuals = [mm.Individual(params_ind, mm.FloryHuggins(chi)) for chi in chis]

    # store some of the results
    data = {
        "input_file": input_path,
        "parameters": result.parameters,
    }

    angles, c_solvent, phi_data = [], [], []
    enriched = {f: [] for f in enriched_fraction}
    diagnostics = {"reached_final_time": 0, "simulation_aborted": 0, "t_final": []}
    for individual in tqdm(individuals):
        # extract variables and parameters
        mixture = individual.dynamics.mixture
        evolve_params = {
            "t_range": individual.parameters["simulation_t_range"],
            "dt": individual.parameters["simulation_dt"],
            "interval": 1.0,
            "tolerance": individual.parameters["equilibrium_tolerance"],
            "progress": False,
            "save_intermediate_data": False,
        }

        # iterate over different initial conditions
        for _ in range(individual.parameters["repetitions"]):
            # choose a new initial composition
            mixture.set_random_composition(
                dist=individual.parameters["composition_distribution"],
                sigma=individual.parameters["composition_sigma"],
            )

            # run simulation
            try:
                ts, result = individual.dynamics.evolve(**evolve_params)
            except RuntimeError:
                # simulation could not finish
                diagnostics["simulation_aborted"] += 1
            else:
                # finalize
                if np.isclose(ts, evolve_params["t_range"]):
                    diagnostics["reached_final_time"] += 1
                diagnostics["t_final"].append(ts)

                angles.append(result.composition_angles())
                phis = result.get_clusters()
                c_solvent.append(1 - phis.sum(axis=1))
                phi_data.append(phis)
                for f in enriched_fraction:
                    enriched[f].append(result.count_enriched_components(f))

    # add cluster analysis
    max_clusters = max(len(phis) for phis in phi_data)
    phi_data = np.concatenate(phi_data)
    clusters, cluster_dist = cluster.vq.kmeans(phi_data, max_clusters)
    data["phi_clusters"] = clusters
    data["phi_cluster_distance"] = float(cluster_dist)

    # add other results
    data["angles"] = np.concatenate(angles)
    data["c_solvent"] = np.concatenate(c_solvent)
    data["diagnostics"] = diagnostics
    for k, v in enriched.items():
        data[f"enriched_{k}"] = np.concatenate(v)

    return data
