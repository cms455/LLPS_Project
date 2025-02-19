#!/usr/bin/env python3 -m modelrunner
"""
This script takes optimized matrices (read in from the results of an optimization run)
and calculates flexibility statistics, which measure how quickly the interaction matrix
can be adopted to another target phase count.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

import time
from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from modelrunner import ModelBase, Parameter, Result

import multicomp as mm


class EvolutionExperiment(ModelBase):
    """simple model class handling an evolution experiment"""

    # compile list of all parameters that this model supports
    parameters_default = [
        Parameter("input_path", None, str, "Path to the input file"),
        Parameter("target_emulsion_size", 2, float, "Evolution goal"),
        Parameter("num_generations", 2, int, "Generation count"),
        Parameter(
            "write_intermediate_data",
            True,
            bool,
            "Write result file at each time step to facilitate early analysis",
        ),
    ]

    def __init__(self, *args, **kwargs):
        """initialize the parameters of the model"""
        super().__init__(*args, **kwargs)
        self.trajectory = []  # collect temporal information of the simulation

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """return the trajectory information of each tracked element"""
        # encode data in a more useful format
        return {
            key: np.array([item[key] for item in self.trajectory])
            for key in self.trajectory[0]
        }

    def __call__(self):
        """run the model"""
        # read input file
        result = Result.from_file(self.parameters["input_path"])
        try:
            chis = np.array(list(result.result["final_chis"].values()))
        except TypeError:
            print(result.result)
            raise

        # extract parameters for the individual and the population
        params_ind = {
            k: v.default_value
            for k, v in mm.Individual.get_parameters(sort=False).items()
        }
        params_pop = {
            k: v.default_value
            for k, v in mm.Population.get_parameters(sort=False).items()
        }

        for key, value in result.parameters.items():
            if key in params_ind:
                params_ind[key] = value
            if key in params_pop:
                params_pop[key] = value

        # construct the population from the data in the input
        individuals = [mm.Individual(params_ind, mm.FloryHuggins(chi)) for chi in chis]
        population = mm.Population(individuals, params_pop)

        # store some diagnostic information
        diagnostics = {"time_evolve": 0.0, "time_writing_intermediates": 0.0}
        t_start_total = time.monotonic()

        # run the simulation for many generations
        for _ in tqdm(range(self.parameters["num_generations"])):
            # evolve the population one generation
            t_start = time.monotonic()
            result = population.evolve(self.parameters["target_emulsion_size"])
            diagnostics["time_evolve"] += time.monotonic() - t_start

            # statistics on component numbers
            num_comps = np.array([ind.num_comps for ind in population])
            chi_stat = population.interaction_matrix_stats()
            size_stat = population.emulsion_size_stats(result["emulsions_sizes"])
            data = {
                "emulsion_size_mean": size_stat["mean"].mean(),
                "emulsion_size_std": size_stat["std"].mean(),
                "fitness_mean": result["fitnesses"].mean(),
                "fitness_std": result["fitnesses"].std(),
                "kill_count": result["kill_count"],
                "num_comps_mean": num_comps.mean(),
                "num_comps_std": num_comps.std(),
                # results from after 2021-10-06 should contain data for `norm` and
                # `entries`, while older data only had data on the `norm`. The following
                # construct tries to reflect his and accepts both forms
                "chi_norm_mean": chi_stat.get("norm_mean", chi_stat.get("mean", None)),
                "chi_norm_std": chi_stat.get("norm_std", chi_stat.get("std", None)),
                "chi_entries_mean": chi_stat.get("entries_mean", None),
                "chi_entries_std": chi_stat.get("entries_std", None),
            }
            self.trajectory.append(data)

            if self.parameters["write_intermediate_data"]:
                # write intermediate result file
                t_start = time.monotonic()
                diagnostics.update(population.diagnostics)
                intermediate_data = {
                    "trajectory": self.get_trajectory(),
                    "diagnostics": diagnostics,
                }
                self.write_result(intermediate_data)
                diagnostics["time_writing_intermediates"] += time.monotonic() - t_start

        # collect final results
        diagnostics["time_total"] = time.monotonic() - t_start_total
        diagnostics.update(population.diagnostics)
        final_chis = [ind.free_energy.chis for ind in population]

        return {
            "trajectory": self.get_trajectory(),
            "final_chis": final_chis,
            "diagnostics": diagnostics,
        }
