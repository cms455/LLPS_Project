"""
Module defining the main optimization protocol.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import copy
import time

import numpy as np
from tqdm.auto import tqdm

from modelrunner import ModelBase, Parameter

from .environment import EnvironmentBase
from .individual import IndividualBase
from .population import Population


def get_optimization_model(
    class_indivdual: type[IndividualBase], class_environment: type[EnvironmentBase]
) -> type[ModelBase]:
    """create a model class that can be used to run an optimization

    Args:
        class_indivdual:
            A subclass of :class:`IndividualBase` defining the population individuals
        class_environment:
            A subclass of :class:`EnvironmentBase` defining the environment

    Returns:
        type :class:`ModelBase`: A subclass of :class:`ModelBase` that implements the
            optimization with the classes specified as arguments.
    """

    class OptimizationModel(ModelBase):
        """simple model class handling an optimization experiment"""

        # compile list of all parameters that this model supports
        parameters_default = (
            [
                Parameter(
                    "output_mode",
                    "auto",
                    str,
                    "The file mode with which the storage is accessed, which "
                    "determines the allowed operations. Common options are 'read', "
                    "'full', 'append', and 'truncate'. The special value 'auto' bases "
                    "the mode on the parameter `write_intermediate_data`",
                ),
                Parameter("num_individuals", 2, int, "Population size"),
                Parameter("num_generations", 2, int, "Generation count"),
                Parameter(
                    "chi_write_frequency",
                    -1,
                    int,
                    "At what (generational) period are the interaction matrices "
                    "written to the output. Negative numbers imply only the final "
                    "state is stored.",
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
            + class_indivdual._parameters_default_full
            + Population._parameters_default_full
            + class_environment._parameters_default_full
        )

        def __init__(self, *args, **kwargs):
            """initialize the parameters of the model"""
            super().__init__(*args, **kwargs)
            self.mode = self.parameters["output_mode"]
            if self.mode == "auto":
                if self.parameters["write_intermediate_data"]:
                    self.mode = "full"
                else:
                    self.mode = "truncate"

            # extract parameters for the individual and the population
            params_ind = class_indivdual.get_parameters(sort=False)
            params_pop = Population.get_parameters(sort=False)
            params_env = class_environment.get_parameters(sort=False)

            for key, value in self.parameters.items():
                if key in params_ind:
                    params_ind[key] = value
                if key in params_pop:
                    params_pop[key] = value
                if key in params_env:
                    params_env[key] = value

            # initialize the optimization class
            individuals = [
                class_indivdual(parameters=params_ind)
                for _ in range(self.parameters["num_individuals"])
            ]
            self.population = Population(individuals, params_pop)
            self.environment = class_environment(params_env)

            self.trajectory = []  # collect temporal information of the simulation

        def get_trajectory(self) -> dict[str, np.ndarray]:
            """return the trajectory information of each tracked element"""
            # encode data in a more useful format
            return {
                key: np.array([item[key] for item in self.trajectory if key in item])
                for key in self.trajectory[0]
            }

        def __call__(self):
            """run the model"""
            # store some diagnostic information
            chi_write_frequency = self.parameters["chi_write_frequency"]
            diagnostics = {"time_evolve": 0.0, "time_writing_intermediates": 0.0}
            t_start_total = time.monotonic()

            # run the simulation for many generations
            for generation in tqdm(range(self.parameters["num_generations"])):
                # evolve the population one generation
                t_start = time.monotonic()
                result = self.environment.evolve(self.population)
                diagnostics["time_evolve"] += time.monotonic() - t_start

                # statistics on component numbers
                num_comps = np.array([ind.num_comps for ind in self.population])
                chi_stat = self.population.interaction_matrix_stats()
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
                    "chi_norm_mean": chi_stat.get("norm_mean", None),
                    "chi_norm_std": chi_stat.get("norm_std", None),
                    "chi_entries_mean": chi_stat.get("entries_mean", None),
                    "chi_entries_std": chi_stat.get("entries_std", None),
                }

                # also store the individual matrices
                if chi_write_frequency > 0 and generation % chi_write_frequency == 0:
                    data["chi_matrices"] = copy.deepcopy(
                        self.population.interaction_matrices
                    )

                self.trajectory.append(data)

                if (
                    self.parameters["write_intermediate_data"]
                    and self.output is not None
                ):
                    # write intermediate result file
                    t_start = time.monotonic()
                    diagnostics.update(self.population.diagnostics)
                    intermediate_data = {
                        "trajectory": self.get_trajectory(),
                        "diagnostics": diagnostics,
                    }
                    result = self.get_result(intermediate_data)
                    self.write_result(result)
                    diagnostics["time_writing_intermediates"] += (
                        time.monotonic() - t_start
                    )

            # collect final results
            diagnostics["time_total"] = time.monotonic() - t_start_total
            diagnostics.update(self.population.diagnostics)
            final_chis = [ind.free_energy.chis for ind in self.population]

            return {
                "trajectory": self.get_trajectory(),
                "final_chis": final_chis,
                "diagnostics": diagnostics,
            }

    return OptimizationModel
