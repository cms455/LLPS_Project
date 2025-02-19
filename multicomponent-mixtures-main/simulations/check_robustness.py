#!/usr/bin/env python3 -m modelrunner
"""
This script takes optimized matrices (read in from the results of an optimization run)
and calculates how robust the performance is with respect to removal or duplication of
components in the optimized matrices.

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
    result_path: str,  #  path to the result file from which the data is read
    modification: str = "all",
):
    """determine the robustness of individuals against gene modifications

    Args:
        result_path (str):
            Path to result file where information about individual is stored
        modification (str):
            What modification is employed to test robustness. Possible values are
            'loss' and 'duplication' as well as 'all', in which case both modifications
            are tried.
    """
    # read result
    result = Result.from_file(result_path)
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

    # construct the population from the data in the input
    individuals = [mm.Individual(params_ind, mm.FloryHuggins(chi)) for chi in chis]
    population = mm.Population(individuals, params_pop)
    individual = individuals[0]  # pick a random individual to modify below

    num_phases = individual.dynamics.mixture.num_phases
    try:
        target_size = result.parameters["target_emulsion_size"]
    except KeyError:
        print(result.parameters.keys())
        raise

    def _get_modification_dataset(modification: str = "loss"):
        """helper function determining the data for a particular modification"""
        dataset = []
        for chi in tqdm(chis):
            # check dropping each component
            for n in range(len(chi)):
                # reset interaction energies
                individual.free_energy.chis = chi

                # modify interaction energies
                if modification == "loss":
                    individual.free_energy.remove_component(n, inplace=True)
                    assert individual.num_comps == len(chi) - 1
                elif modification == "duplication":
                    individual.free_energy.duplicate_component(n, inplace=True)
                    assert individual.num_comps == len(chi) + 1
                else:
                    raise ValueError(f"Unknown modification `{modification}`")

                # set up the initial state of the mixture
                mix = mm.MultiphaseSystem.from_random_composition(
                    individual.free_energy, num_phases
                )
                individual.dynamics.mixture = mix

                # estimate emulsion size (phase count distribution)
                emulsion_sizes = individual.estimate_emulsion_size(progress=False)

                # calculate the fitness based on this
                fitness = population.get_population_fitness(emulsion_sizes, target_size)

                # collect data
                dataset.append(
                    {
                        "chi": chi.tolist(),
                        "emulsion_sizes": emulsion_sizes.tolist(),
                        "fitness": float(fitness),
                    }
                )

        return dataset

    # store some of the results
    data = {
        "input_file": result_path,
        "parameters": result.parameters,
        "modification": modification,
    }

    # do the actual calculation
    if modification in {"loss", "all"}:
        data["data_loss"] = _get_modification_dataset("loss")
    if modification in {"duplication", "all"}:
        data["data_duplication"] = _get_modification_dataset("duplication")

    # add additional results
    try:
        trajectory = result.result["trajectory"]
        data["fitness_mean_init"] = float(trajectory["fitness_mean"][-1])
        data["fitness_std_init"] = float(trajectory["fitness_std"][-1])
    except Exception:
        print("Could not find the original fitness")

    return data
