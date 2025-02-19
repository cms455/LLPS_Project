#!/usr/bin/env python3 -m modelrunner
"""
This script takes optimized matrices (read in from the results of an optimization run)
and calculates how robust the performance is with respect to perturbations of the
interaction matrices.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

import numpy as np
from tqdm.auto import tqdm

from modelrunner import Result, make_model

import multicomp as mm


@make_model
def run(
    input_path: str,  #  path to the result file from which the data is read
    strategy: str = "all_entries",  # mutation strategy
    pert_min: float = 0,  # minimal perturbation
    pert_max: float = 10,  # maximal perturbation
    pert_count: int = 3,  # number of perturbations
    pert_log: bool = False,  # whether to use a logscale
):
    """determine the composition statistics of states

    Args:
        result_path (str):
            Path to result file where information about individual is stored
    """
    if pert_log:
        perts = np.geomspace(pert_min, pert_max, pert_count)
    else:
        perts = np.linspace(pert_min, pert_max, pert_count)

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
    params_pop = {
        k: v.default_value for k, v in mm.Population.get_parameters(sort=False).items()
    }
    for key, value in result.parameters.items():
        if key in params_ind:
            params_ind[key] = value
        if key in params_pop:
            params_pop[key] = value

    # set parameters for the current perturbation strategy
    params_ind["chi_limit"] = "none"
    params_ind["mutation_strategy"] = strategy
    params_ind["gene_dup_rate"] = 0
    params_ind["gene_loss_rate"] = 0

    # construct the population from the data in the input
    individuals = [mm.Individual(params_ind, mm.FloryHuggins(chi)) for chi in chis]
    population = mm.Population(individuals, params_pop)

    # store some of the results
    data = {
        "input_file": input_path,
        "parameters": result.parameters,
    }
    target_phase_count = result.parameters["target_emulsion_size"]

    performance = np.full((len(individuals), pert_count), np.nan)
    diagnostics = {"reached_final_time": 0, "simulation_aborted": 0, "t_final": []}
    for i, individual in enumerate(tqdm(individuals)):
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
        for j, pert in enumerate(perts):
            individual.parameters["mutation_size"] = pert

            emulsion_sizes = []
            for _ in range(individual.parameters["repetitions"]):
                # reset interaction energies
                individual.free_energy.chis[:] = chis[i]
                individual.mutate()
                assert individual.free_energy.chis is mixture.free_energy.chis

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

                    dist = individual.parameters["cluster_threshold"]
                    emulsion_sizes.append(result.count_clusters(dist=dist))

            performance[i, j] = population.get_population_fitness(
                emulsion_sizes=np.bincount(emulsion_sizes),
                target_size=target_phase_count,
            )

    # add other results
    data["perturbations"] = perts
    data["performance"] = performance

    return data
