#!/usr/bin/env python3 -m modelrunner
"""
This script optimizes interaction matrices with respect to the phase count.

This is a legacy script, which shows a detailed implementation. Newer codes should use
`optimize_phase_count.py` instead.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

import copy
import time
from typing import Dict

import numpy as np
from tqdm.auto import tqdm

from modelrunner import ModelBase, Parameter, submit_job

from multicomp import FullChiIndividual, Population, TargetPhaseCountEnvironment


class EvolutionExperiment(ModelBase):
    """simple model class handling an evolution experiment"""

    # compile list of all parameters that this model supports
    parameters_default = (
        [
            Parameter("num_individuals", 2, int, "Population size"),
            Parameter("num_generations", 2, int, "Generation count"),
            Parameter(
                "chi_write_frequency",
                -1,
                int,
                "At what (generational) period are the interaction matrices written to "
                "the output. Negative numbers imply only the final state is stored.",
            ),
            Parameter(
                "write_intermediate_data",
                True,
                bool,
                "Write result file at each time step to facilitate early analysis",
            ),
            Parameter(
                "ensemble_id",
                0,
                int,
                "An integer allowing to distinguish different runs in an ensemble. "
                "This value does not affect the simulation, but is stored here for "
                "convenience.",
            ),
        ]
        + FullChiIndividual._parameters_default_full
        + Population._parameters_default_full
        + TargetPhaseCountEnvironment._parameters_default_full
    )

    def __init__(self, *args, **kwargs):
        """initialize the parameters of the model"""
        super().__init__(*args, **kwargs)
        self.trajectory = []  # collect temporal information of the simulation

    def get_trajectory(self) -> Dict[str, np.ndarray]:
        """return the trajectory information of each tracked element"""
        # encode data in a more useful format
        return {
            key: np.array([item[key] for item in self.trajectory if key in item])
            for key in self.trajectory[0]
        }

    def __call__(self):
        """run the model"""
        # extract parameters for the individual and the population
        params_ind = FullChiIndividual.get_parameters(sort=False)
        params_pop = Population.get_parameters(sort=False)
        params_env = TargetPhaseCountEnvironment.get_parameters(sort=False)
        for key, value in self.parameters.items():
            if key in params_ind:
                params_ind[key] = value
            if key in params_pop:
                params_pop[key] = value
            if key in params_env:
                params_env[key] = value

        # initialize the population
        individuals = [
            FullChiIndividual(params_ind)
            for _ in range(self.parameters["num_individuals"])
        ]
        population = Population(individuals, params_pop)
        environment = TargetPhaseCountEnvironment(params_env)

        # store some diagnostic information
        chi_write_frequency = self.parameters["chi_write_frequency"]
        diagnostics = {"time_evolve": 0.0, "time_writing_intermediates": 0.0}
        t_start_total = time.monotonic()

        # run the simulation for many generations
        for generation in tqdm(range(self.parameters["num_generations"])):
            # evolve the population one generation
            t_start = time.monotonic()
            result = environment.evolve(population)
            diagnostics["time_evolve"] += time.monotonic() - t_start

            # statistics on component numbers
            num_comps = np.array([ind.num_comps for ind in population])
            chi_stat = population.interaction_matrix_stats()
            # size_stat = population.emulsion_size_stats(result["emulsions_sizes"])
            data = {
                # "emulsion_size_mean": size_stat["mean"].mean(),
                # "emulsion_size_std": size_stat["std"].mean(),
                "fitness_mean": result["fitnesses"].mean(),
                "fitness_std": result["fitnesses"].std(),
                "kill_count": result["kill_count"],
                "lineage": result["lineage"],
                "num_comps_mean": num_comps.mean(),
                "num_comps_std": num_comps.std(),
                # results from after 2021-10-06 should contain data for `norm` and
                # `entries`, while older data only had data on the `norm`. The following
                # construct tries to reflect this and accepts both forms
                "chi_norm_mean": chi_stat.get("norm_mean", chi_stat.get("mean", None)),
                "chi_norm_std": chi_stat.get("norm_std", chi_stat.get("std", None)),
                "chi_entries_mean": chi_stat.get("entries_mean", None),
                "chi_entries_std": chi_stat.get("entries_std", None),
            }

            # also store the individual matrices
            if chi_write_frequency > 0 and generation % chi_write_frequency == 0:
                data["chi_matrices"] = copy.deepcopy(population.interaction_matrices)

            self.trajectory.append(data)

            if self.parameters["write_intermediate_data"] and self.output is not None:
                # write intermediate result file
                t_start = time.monotonic()
                diagnostics.update(population.diagnostics)
                intermediate_data = {
                    "trajectory": self.get_trajectory(),
                    "diagnostics": diagnostics,
                }
                self.write_result(output=None, result=intermediate_data)
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


if __name__ == "__main__":
    submit_job(__file__, output="result.hdf5", method="qsub")
