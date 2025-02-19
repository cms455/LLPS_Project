"""
Module describing a population of individuals.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import functools
import multiprocessing as mp
from statistics import mean
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from modelrunner.parameters import Parameter, Parameterized

from .individual import IndividualBase


class Population(Parameterized):
    """represents a population of diverse individuals"""

    parameters_default = [
        Parameter("num_processes", 1, int, "Number of parallel processes to use"),
    ]

    def __init__(
        self,
        individuals: list[IndividualBase],
        parameters: dict[str, Any] = None,
        *,
        strict: bool = True,
    ):
        """
        Args:
            individuals: List of all individuals in the population
            parameters: Additional parameters affecting the population
            strict (bool): Whether parameters are strictly enforced
        """
        super().__init__(parameters, strict=strict)
        if not all(isinstance(ind, IndividualBase) for ind in individuals):
            raise TypeError("Individuals must be of type IndividualBase")
        # check whether all individuals are of the same type

        self.individuals = individuals
        self.diagnostics = {"reached_final_time": 0, "simulation_aborted": 0}

    def __len__(self):
        return len(self.individuals)

    def __getitem__(self, index):
        return self.individuals[index]

    def __iter__(self):
        return iter(self.individuals)

    @property
    def interaction_matrices(self) -> list[np.ndarray]:
        """return all interaction matrices of the individuals"""
        return [individual.free_energy.chis for individual in self]

    def interaction_matrix_stats(self) -> dict[str, float]:
        """return statistics of the norm of the interaction matrices of individuals"""
        chi_norms, chi_entries = [], []
        for individual in self:
            chis = individual.free_energy.chis
            # determine the norm of the chi matrix
            chi_norms.append(np.linalg.norm(chis))
            # extract entries of upper triangle
            chi_entries.append(individual.free_energy.independent_entries)

        return {
            "norm_mean": np.mean(chi_norms),
            "norm_std": np.std(chi_norms),
            "entries_mean": mean(np.mean(entries) for entries in chi_entries),
            "entries_std": mean(np.std(entries) for entries in chi_entries),
        }

    def get_stats(
        self, quantities: str | Iterable[str], progress: bool = False
    ) -> list[dict[str, Any]]:
        """determine the emulsion size distribution of all individuals

        Args:
            quantities (str or list of str):
                Quantities that are included in the ensemble. This supports the same
                values as
                :meth:`~multicomp.evolution.individual.IndividualBase.get_ensemble`.
            progress (bool):
                Whether to show a progress bar in serial calculations

        Returns:
            list: a dictionary for each individual that contains the ensemble statistics
        """
        if isinstance(quantities, str):
            quantities_set = {quantities}
        else:
            quantities_set = set(quantities)

        # reset diagnostic information
        for individual in self:
            individual.diagnostics["reached_final_time"] = 0
            individual.diagnostics["simulation_aborted"] = 0

        # determine the individual distribution
        if self.parameters["num_processes"] > 1:
            # use multiprocessing

            get_ensemble = functools.partial(
                self[0].__class__.get_ensemble,
                quantities=quantities_set,
                progress=False,
            )
            with mp.Pool(self.parameters["num_processes"], maxtasksperchild=1) as pool:
                population_stats = pool.map(get_ensemble, self)

        else:
            # use single process
            population_stats = [
                individual.get_ensemble(quantities=quantities_set, progress=False)
                for individual in tqdm(self, disable=not progress)
            ]

        # collect diagnostic information
        for individual in self:
            for key in ["reached_final_time", "simulation_aborted"]:
                self.diagnostics[key] += individual.diagnostics[key]

        return population_stats

    def replace_asexually(
        self, kill_idx: np.ndarray, *, weights: np.ndarray = None
    ) -> np.ndarray:
        """replace some individuals with reproduced ones

        Args:
            kill_idx (:class:`np.ndarray`): Indices of individuals that will be replaced
            weights (:class:`np.ndarray`): Weights affecting the replacement choice

        Returns:
            :class:`np.ndarray`: Indices describing the lineage
        """
        lineage = np.arange(len(self), dtype=int)

        if len(kill_idx) == 0:
            # nothing to do
            return lineage

        # determine the individuals that are kept
        keep_idx = np.array([i for i in range(len(self)) if i not in kill_idx])

        # weight reproduction of surviving individuals by fitness
        if weights is None:
            weights_keep = None
        else:
            weights_keep = weights[keep_idx] / weights[keep_idx].sum()
        for i in kill_idx:
            # weighted choice of a surviving individual
            j = np.random.choice(keep_idx, p=weights_keep)
            self.individuals[i] = self.individuals[j].copy()
            lineage[i] = j

        return lineage

    def replace_sexually(
        self, kill_idx: np.ndarray, *, weights: np.ndarray = None
    ) -> np.ndarray:
        """replace some individuals by combining two others

        Args:
            kill_idx (:class:`np.ndarray`): Indices of individuals that will be replaced
            weights (:class:`np.ndarray`): Weights affecting the replacement choice

        Returns:
            :class:`np.ndarray`: Indices describing the lineage
        """
        lineage = np.c_[np.arange(len(self)), np.arange(len(self))].astype(int)

        if len(kill_idx) == 0:
            # nothing to do
            return lineage  # type: ignore

        # determine the individuals that are kept
        keep_idx = np.array([i for i in range(len(self)) if i not in kill_idx])
        parents = self.individuals.copy()

        if len(keep_idx) == 1:
            # we cannot really do sexual reproduction, so just copy only individual
            only_parent = keep_idx[0]

            for i in kill_idx:
                self.individuals[i] = parents[only_parent].copy()

            lineage = np.full((len(self), 2), only_parent, dtype=int)

        else:
            # combine interaction matrices of two individuals

            # weight reproduction of surviving individuals by fitness
            if weights is None:
                weights_rep = None
            else:
                weights_keep = [
                    weights[i] for i in range(len(self)) if i not in kill_idx
                ]
                weights_rep = np.array(weights_keep) / sum(weights_keep)
            for i in kill_idx:
                # weighted choice of a surviving individual
                i1, i2 = np.random.choice(keep_idx, 2, replace=False, p=weights_rep)
                # take first individual as basis and replace some parts using second one
                self.individuals[i] = parents[i1].crossover(parents[i2])
                # save lineage
                lineage[i] = (i1, i2)

        return lineage  # type: ignore
