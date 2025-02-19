"""
Environments determine the fitness of individuals and thus the optimization goals. 

.. autosummary::
   :nosignatures:

   ~TargetPhaseCountEnvironment
   ~PartitioningEnvironment

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import Any

import numpy as np

from modelrunner.parameters import Parameter, Parameterized

from .population import Population


class EnvironmentBase(Parameterized, metaclass=ABCMeta):
    """abstract base class describing an environment"""

    parameters_default = [
        Parameter(
            "kill_method",
            "kill_unfit_fraction",
            str,
            "Chooses the method for removing individuals:"
            " 'kill_by_fitness': Kill individuals randomly according to their fitness"
            " 'kill_unfit_fraction': Replace a fraction of most unfit individuals",
        ),
        Parameter(
            "evolution_pressure",
            0.5,
            float,
            "Evolutionary pressure that determines how the fitness of an individual "
            "affects its kill rate. Only used when `kill_method == 'kill_by_fitness'`.",
        ),
        Parameter(
            "replace_fraction",
            0.1,
            float,
            "Fraction of individuals to replace each generation. This gives the exact "
            "fraction when `kill_method == 'kill_unfit_fraction'` and the maximal "
            "fraction when `kill_method == 'kill_by_fitness'`.",
        ),
        Parameter(
            "replace_method",
            "asexual",
            str,
            "Method for replacing individuals: either 'sexual' or 'asexual'",
        ),
    ]

    def __init__(self, parameters: dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
        """
        super().__init__(parameters)
        self.diagnostics: dict[str, Any] = {}

    @abstractmethod
    def get_population_fitness(self, population: Population) -> np.ndarray:
        pass

    def _kill_by_fitness(self, fitnesses: np.ndarray) -> np.ndarray:
        """kill individuals according to their fitness

        Args:
            fitnesses (:class:`np.ndarray`): Fitness of all individuals

        Returns:
            :class:`np.ndarray`: Indices of individuals to be killed
        """
        # determine the rates with which individuals are killed
        replace_fraction = self.parameters["replace_fraction"]
        evolution_pressure = self.parameters["evolution_pressure"]
        kill_rate = replace_fraction * (1 - fitnesses) ** evolution_pressure
        # determine which individuals are killed
        kill_mask = np.random.rand(len(fitnesses)) < kill_rate

        # make sure at least one is killed and at least one survives
        if np.all(kill_mask):
            # the fittest individual survives
            kill_mask[np.argmax(fitnesses)] = False
        elif not np.any(kill_mask):
            # the unfittest individual is killed
            kill_mask[np.argmin(fitnesses)] = True

        return np.flatnonzero(kill_mask)

    def _kill_unfit_fraction(self, fitnesses: np.ndarray) -> np.ndarray:
        """obtain most unfit individuals

        Args:
            fitnesses (:class:`np.ndarray`): Fitness of all individuals

        Returns:
            :class:`np.ndarray`: Indices of individuals to be killed
        """
        # determine the number of individuals that need to be replaced
        kill_count = round(self.parameters["replace_fraction"] * len(fitnesses))
        # make sure at least one is killed and at least one survives
        kill_count = int(np.clip(kill_count, 1, len(fitnesses) - 1))
        # kill least fit individuals
        return np.argsort(fitnesses)[:kill_count]

    def evolve(self, population: Population) -> dict[str, Any]:
        """evolve a population one generation

        Args:
            population (:class:`Population`): The population to evolve

        Returns:
            dict: Some information about this one step of evolution
        """
        # mutate all individuals
        for individual in population:
            individual.mutate()

        # get associated fitnesses
        fitnesses = self.get_population_fitness(population)

        # determine which individuals to kill
        if self.parameters["kill_method"] == "kill_by_fitness":
            kill_idx = self._kill_by_fitness(fitnesses)
        elif self.parameters["kill_method"] in {"kill_unfit", "kill_unfit_fraction"}:
            kill_idx = self._kill_unfit_fraction(fitnesses)
        else:
            raise ValueError(f"Unknown kill method {self.parameters['kill_method']}")
        kill_count = len(kill_idx)  # number of individuals that will be killed

        # replace these individuals by reproduced ones
        replace_method = self.parameters["replace_method"]
        if kill_count == len(population):
            self._logger.warning("Spawn random individuals since all were killed")
            for individual in population:
                individual.randomize()
            if replace_method == "sexual":
                shape: tuple[int, ...] = (len(population), 2)
            else:
                shape = (len(population),)
            lineage = np.full(shape, np.nan)

        elif replace_method == "asexual":
            lineage = population.replace_asexually(kill_idx, weights=fitnesses)

        elif replace_method == "sexual":
            lineage = population.replace_sexually(kill_idx, weights=fitnesses)

        else:
            raise ValueError(f"Unknown replace method {replace_method}")

        return {
            "fitnesses": fitnesses,
            "kill_count": kill_count,
            "lineage": lineage,
        }


class TargetPhaseCountEnvironment(EnvironmentBase):
    r"""Environment whose fitness is based on a constant target phase count

    The performance of an individual in this environment depends only on the number of
    phases that form. The individual performance is defined

    .. math::
        g = \sum_K P(K) \exp\left[-\frac{(K-K_*)^2}{2w^2}\right]

    The performance peaks when all systems exhibit phase counts :math:`K` close to the
    desired phase count :math:`K_*`, which is set by `target_phase_count`. The tolerance
    :math:`w`, set by `phase_count_tolerance`, determines how strongly deviations are
    punished.
    """

    parameters_default = [
        Parameter("target_phase_count", 3, float, "Target phase count"),
        Parameter(
            "phase_count_tolerance", 1.0, float, "Tolerance in meeting target count"
        ),
    ]

    def _fitness_function(self, phase_counts: np.ndarray) -> np.ndarray:
        """return the fitness associated with individual phase counts

        Args:
            phase_counts (:class:`np.ndarray`): One or many phase counts

        Returns:
            The performance associated with each phase count
        """
        target_count = self.parameters["target_phase_count"]
        tolerance = self.parameters["phase_count_tolerance"]
        arg = (np.asanyarray(phase_counts) - target_count) / tolerance
        return np.exp(-0.5 * arg**2)  # type: ignore

    def get_fitness(self, phase_counts: np.ndarray) -> np.ndarray:
        """calculates the fitness associated with a particular set of phase counts

        Args:
            phase_counts (:class:`np.ndarray`): One or many phase counts

        Returns:
            Fitness/performance associated with the phase count, which are normalized to
            lie between 0 and 1. The returned array will have one dimension less than
            the input.
        """
        return self._fitness_function(phase_counts).mean(axis=-1)  # type: ignore

    def get_population_fitness(self, population: Population, **kwargs) -> np.ndarray:
        """return a fitness for each member of a population

        Args:
            population (:class:`Population`): The individuals whose fitness is requested

        Returns:
            :class:`np.ndarray`: Fitness of all individuals. These are normalized to lie
            between 0 and 1.
        """
        # determine the statistics of the population
        stats = population.get_stats(["phase_counts"], **kwargs)

        # determine fitness of each individual
        fitnesses = np.empty(len(population))
        for i, stat in enumerate(stats):
            counts = np.asarray(stat["phase_counts"])
            weights = self._fitness_function(counts)
            fitnesses[i] = np.mean(weights)
        return fitnesses

    def plot_fitness_function(self):
        """visualizes how a particular emulsion sizes affects the fitness"""
        import matplotlib.pyplot as plt

        target_size = self.parameters["target_phase_count"]
        size_max = target_size + 3 * self.parameters["phase_count_tolerance"]
        sizes = np.linspace(0, size_max)
        weights = self._fitness_function(sizes)
        plt.plot(sizes, weights)


class PartitioningEnvironment(EnvironmentBase):
    r"""Environment rewarding meeting target phase count and strong partitioning

    The performance :math:`g` of an individual in this environment depends on two
    aspects, the number :math:`K` of phases that form and the enrichment of components
    in phases. These aspects are quantified using performances :math:`g_\mathrm{count}`
    and :math:`g_\mathrm{partition}`, respectively, which are combined using the
    weighting parameter `fitness_weight` (:math:`\alpha`),

    .. math::
        \begin{align}
        g_\mathrm{count} &= \sum_K P(K) \exp\left[-\frac{(K-K_*)^2}{2w^2}\right]
        \\
        g_\mathrm{partition} &= \frac{1}{L} \sum_{i=1}^L
            \max_n \left(\phi_i^{(n)}  - \langle \phi_i^{(m)}\rangle_{m\neq n}\right)
        \\
        g &= (1 - \alpha) g_\mathrm{count} + \alpha g_\mathrm{partition} 
        \end{align}

    Here, the first performance peaks when all systems exhibit phase counts :math:`K`
    close to the desired phase count :math:`K_*`, which is set by `target_phase_count`.
    The tolerance :math:`w`, set by `phase_count_tolerance`, determines how strongly
    deviations are punished. The second performance peaks when the select subset of
    :math:`L` components partition strongly into a phase. Here, the partitioning is
    defined as the difference of the fraction in a particular phase minus the average
    fraction in all other phases. The number :math:`L` of components whose enrichment is
    controlled is set by the parameter `enriched_components`.
    """

    parameters_default = [
        Parameter("target_phase_count", 3, float, "Target phase count"),
        Parameter(
            "phase_count_tolerance", 1.0, float, "Tolerance in meeting target count"
        ),
        Parameter(
            "enriched_components",
            [],
            list,
            "List of component (indices) that need to be enriched",
        ),
        Parameter(
            "fitness_weight",
            0.5,
            float,
            "Weighting of the target phase count requirement vs the enriched components "
            "requirements. The value needs to be between 0 and 1. 0 implies that only "
            "the composition contributes, while 1 indicates that only the phase count "
            "is considered.",
        ),
    ]

    def __init__(self, parameters: dict[str, Any] = None):
        """
        Args:
            parameters (dict):
                A dictionary of parameters to change the defaults. The allowed
                parameters can be obtained from
                :meth:`~Parameterized.get_parameters` or displayed by calling
                :meth:`~Parameterized.show_parameters`.
        """
        super().__init__(parameters)
        if len(self.parameters["enriched_components"]) == 0:
            self._logger.warning("Did not specify any components to enrich")
        else:
            values = self.parameters["enriched_components"]
            self.parameters["enriched_components"] = np.array([int(v) for v in values])

    def get_population_fitness(
        self, population: Population, *, weight: float = None, **kwargs
    ) -> np.ndarray:
        """return a number between 0 and 1 determining fitness

        Args:
            population (:class:`Population`):
                The individuals whose fitness is requested
            weight (float):
                Parameter that determines the weighting of the target phase count
                requirement vs the enriched components requirements. The value needs to
                be between 0 and 1. 0 implies that only the composition contributes,
                while 1 indicates that only the phase count is considered. If this value
                is `None`, the parameter `fitness_weight` from the `parameters`
                attribute is used.

        Returns:
            :class:`np.ndarray`: Fitness of all individuals. These are normalized to lie
            between 0 and 1.
        """
        # determine the statistics of the population
        stats = population.get_stats(["phase_counts", "phis"], **kwargs)

        # read parameters
        target_count = self.parameters["target_phase_count"]
        phase_count_tol = self.parameters["phase_count_tolerance"]
        enriched_components = self.parameters["enriched_components"]
        if weight is None:
            weight = self.parameters["fitness_weight"]

        # determine fitness of each individual
        fitnesses = np.empty(len(population))
        for i, ind_stat in enumerate(stats):  # iterate over individuals
            # determine phase count fitness
            counts = np.asarray(ind_stat["phase_counts"])
            arg = (counts - target_count) / phase_count_tol
            fitness_count = np.mean(np.exp(-0.5 * arg**2))

            if len(enriched_components) > 0:
                # determine maximal partition coefficient
                result = []
                for phis in ind_stat["phis"]:  # iterate over ensemble
                    # phis is now a 2d array of compositions (phases x component)
                    phis_sel = phis[:, enriched_components]
                    K = len(phis_sel)  # phase count of the individual sample
                    if K == 1:  # no phase separation
                        result.append(np.zeros(len(enriched_components)))
                    else:
                        # calculate the difference between the fraction in a particular
                        # phase and the average of all other phases
                        partition = (K * phis_sel - phis_sel.sum(axis=0)) / (K - 1)
                        # take the result from the best phase
                        result.append(np.max(partition, axis=0))

                # average the partitioning fitness of all target components
                fitness_partition = np.mean(result)

                # use the weighted average of both fitness
                fitnesses[i] = weight * fitness_count + (1 - weight) * fitness_partition

            else:
                # use only the fitness count since enriched components are not specified
                fitnesses[i] = fitness_count

        return fitnesses
