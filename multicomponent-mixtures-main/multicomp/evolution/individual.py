"""
Module defining individuals and particularly what quantities change under evolution.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import collections
import json
from abc import ABCMeta, abstractmethod
from typing import Any, Iterable

import numpy as np
from tqdm.auto import tqdm

from modelrunner.parameters import NoValue, Parameter, Parameterized

from ..dynamics import RelaxationDynamics
from ..dynamics_field_like import FieldLikeRelaxationDynamics
from ..mixture import MultiphaseSystem, MultiphaseVolumeSystem, get_composition_angles
from ..thermodynamics import FloryHuggins


class IndividualBase(Parameterized, metaclass=ABCMeta):
    """abstract base class describing an individual"""

    parameters_default = [
        # parameters for estimating emulsion size
        Parameter(
            "repetitions",
            3,
            int,
            "Number of repetitions to estimate ensemble statistics",
        ),
        Parameter("num_phases", NoValue, int, "Initial number of phases in mixture"),
        Parameter(
            "composition_distribution",
            "simplex_uniform",
            str,
            "The distribution from which the initial compositions are sampled. Possible "
            "values are `lognormal`, `uniform`, and `simplex_uniform`.",
        ),
        Parameter(
            "composition_sigma",
            1.0,
            float,
            "Width of the initial concentration distribution of the phases. This is "
            "only used when 'composition_distribution' == 'lognormal'.",
        ),
        Parameter("simulation_t_range", 1e4, float, "Maximal simulation time"),
        Parameter("simulation_dt", 0.1, float, "Time step of the simulation"),
        Parameter("equilibrium_tolerance", 1e-5, float, "Steady state rate threshold"),
        Parameter("cluster_threshold", 1e-2, float, "Tolerance for clustering phases"),
        Parameter(
            "relaxation_method",
            RelaxationDynamics.get_parameter_default("method"),
            str,
            "Determines the relaxation method, which can be `explicit`, "
            "`explicit_adaptive`, or `implicit`.",
        ),
        Parameter(
            "cache_dynamics", True, bool, "Determines whether dynamics object is cached"
        ),
        Parameter(
            "dynamics_implementation",
            "standard",
            str,
            "Determines the implementation of relaxation dynamic, which can be `standard`, "
            "or `field_like`. Note that when `field_like` is used, options `relaxation_method`"
            "will be ignored, and `simulation_dt` needs to be 1.",
        ),
        Parameter(
            "dynamics_parameters",
            {},
            object,
            "Additional parameters for relaxation dynamic. For command line input, this should"
            "be a valid json string with proper escape.",
        ),
    ]

    parameters: dict[str, Any]  # explicit type hint
    free_energy: FloryHuggins

    def __init__(self, parameters: dict[str, Any] = None, *, rng=None):
        """
        Args:
            parameters (dict): Parameters of this individual
            rng: the random number generator
        """
        super().__init__(parameters)

        self.rng = np.random.default_rng() if rng is None else rng
        self.dynamics: RelaxationDynamics | FieldLikeRelaxationDynamics | None = None

        self.diagnostics = {
            "reached_final_time": 0,
            "simulation_aborted": 0,
        }

    @property
    def num_comps(self) -> int:
        """int: number of components in the mixture of this individual"""
        return self.free_energy.num_comps

    @abstractmethod
    def copy(self) -> IndividualBase:
        pass

    @abstractmethod
    def randomize(self) -> None:
        pass

    @abstractmethod
    def mutate(self):
        pass

    def get_ensemble(
        self, quantities: str | Iterable[str], *, progress: bool = True
    ) -> dict[str, Any]:
        """calculates quantities over the ensemble of initial conditions

        Args:
            quantities (str):
                Identifiers of quantities that are included in the result. Possible
                values include `phi_initial`, `phase_counts`, `phis`, `phi_solvent`, and
                `angles`.
            progress (bool): Whether to show a progress bar

        Returns:
            dict: A dictionary with the results requested by `quantities`
        """
        # read parameters
        repetitions = self.parameters["repetitions"]
        num_phases = self.parameters["num_phases"]
        composition_distribution = self.parameters["composition_distribution"]
        composition_sigma = self.parameters["composition_sigma"]
        dynamics_impl = self.parameters["dynamics_implementation"]

        # default simulation parameters
        t_range = self.parameters["simulation_t_range"]
        dt = self.parameters["simulation_dt"]
        tolerance = self.parameters["equilibrium_tolerance"]
        cluster_threshold = self.parameters["cluster_threshold"]

        # check implementation in advance
        if dynamics_impl not in ["standard", "field_like"]:
            raise ValueError(f"Unknown dynamics_implementation `{dynamics_impl}`")

        # prepare simulation
        if not self.parameters["cache_dynamics"] or self.dynamics is None:
            if isinstance(self.parameters["dynamics_parameters"], str):
                json_acceptable_string = self.parameters["dynamics_parameters"].replace(
                    "'", '"'
                )
                try:
                    dynamics_parameters = json.loads(json_acceptable_string)
                except json.decoder.JSONDecodeError as e:
                    raise ValueError(f"{json_acceptable_string} cannot be interpreted as a proper json string") from e
                
            elif isinstance(self.parameters["dynamics_parameters"], dict):
                dynamics_parameters = self.parameters["dynamics_parameters"]
            else:
                temp = self.parameters["dynamics_parameters"]
                raise ValueError(f"{temp} cannot be interpreted as a dict")

            if dynamics_impl == "standard":
                mixture = MultiphaseSystem.from_random_composition(
                    self.free_energy, self.parameters["num_phases"], rng=self.rng
                )
                p = {
                    "method": self.parameters["relaxation_method"],
                    **dynamics_parameters,
                }
                self.dynamics = RelaxationDynamics(mixture, parameters=p)
            elif dynamics_impl == "field_like":
                mixture = MultiphaseVolumeSystem.from_random_composition(
                    self.free_energy,
                    np.ones(self.parameters["num_phases"]),
                    rng=self.rng,
                )
                p = {
                    "initialization_method": "random",
                    **dynamics_parameters,
                }
                self.dynamics = FieldLikeRelaxationDynamics(
                    mixture, rng=self.rng, parameters=p
                )

        if isinstance(quantities, str):
            quantities_set = {quantities}
        else:
            quantities_set = set(quantities)

        diagnostics: dict[str, Any] = {
            "t_final": [],
            "reached_final_time": 0,
            "simulation_aborted": 0,
        }
        result: collections.defaultdict[str, Any] = collections.defaultdict(list)
        for _ in tqdm(range(repetitions), disable=not progress):
            # choose random initial condition
            self.dynamics.mixture.set_random_composition(
                dist=composition_distribution,
                num_phases=num_phases,
                sigma=composition_sigma,
            )
            if dynamics_impl == "field_like":
                # change the composition to all-same, which makes the average volume
                # fractions correctly match the chosen distribution
                self.dynamics.mixture.phis = np.outer(
                    np.ones(self.dynamics.mixture.num_phases),
                    self.dynamics.mixture.phis[0],
                )

            if "phi_initial" in quantities_set:
                phi_initial = self.dynamics.mixture.phis.copy()
                if dynamics_impl == "field_like":
                    self._logger.warn(
                        """
                    You request the "phi_initial" property while "dynamics_implementation"
                    is chosen to be "field_like". Note that field-like dynamics do not respect
                    the initial composition but only the mean value, therefore the recorded
                    data provides no info about the real initial volume fractions. 
                    """
                    )

            # run relaxation dynamics again
            try:
                ts, mixture = self.dynamics.evolve(
                    t_range=t_range,
                    dt=dt,
                    interval=100 * dt,
                    tolerance=tolerance,
                    progress=False,
                )
            except RuntimeError:
                # simulation could not finish
                diagnostics["simulation_aborted"] += 1
            else:
                # finalize
                if np.isclose(ts, t_range):
                    diagnostics["reached_final_time"] += 1
                diagnostics["t_final"].append(ts)

                # obtain cluster data
                phis = mixture.get_clusters(dist=cluster_threshold)

                # store requested result
                if "counts" in quantities_set:
                    result["counts"].append(len(phis))
                if "phase_counts" in quantities_set:
                    result["phase_counts"].append(len(phis))
                if "phis" in quantities_set:
                    result["phis"].append(phis)
                if "phi_solvent" in quantities_set:
                    result["phi_solvent"].append(1 - phis.sum(axis=1))
                if "phi_initial" in quantities_set:
                    result["phi_initial"].append(phi_initial)
                if "angles" in quantities_set:
                    angles = get_composition_angles(phis, squareform=False)
                    result["angles"].extend(angles)

        # store diagnostic information
        self.diagnostics["reached_final_time"] += diagnostics["reached_final_time"]
        self.diagnostics["simulation_aborted"] += diagnostics["simulation_aborted"]

        return dict(result)


class FullChiIndividual(IndividualBase):
    """represents an individual described by an interaction matrix"""

    parameters_default = [
        # parameters for estimating initial agent properties
        Parameter("num_comp_init", NoValue, int, "number of components"),
        Parameter("chi_mean", NoValue, float, "mean value of the interactions"),
        Parameter("chi_std", NoValue, float, "standard deviation of the interactions"),
        Parameter(
            "chi_limit",
            "none",
            str,
            "Method for limiting interaction values. Possible values are `none`, "
            "`norm`, and `clip`.",
        ),
        Parameter("chi_min", -np.inf, float, "minimal value for chi_limit == 'clip'"),
        Parameter("chi_max", np.inf, float, "maximal value for chi_limit == 'clip'"),
        Parameter("chi_norm_max", np.inf, float, "maximal norm if chi_limit == 'norm'"),
        Parameter("randomize_chi_mean", True, bool, "randomize chi_mean each time"),
        Parameter("inert_solvent", True, bool, "whether chi has vanishing diagonal"),
        # parameters for mutating the individual
        Parameter(
            "mutation_strategy",
            "all_entries",
            str,
            "Strategy used to mutate the interaction matrix. Possible values are:"
            " 'single_entry': Add random perturbation to a single entry"
            " 'one_per_component': Add random perturbation to one entry per component"
            " 'all_entries`: Add random perturbation to all entries"
            "The magnitude of the perturbation is set by `mutation_size`",
        ),
        Parameter("mutation_size", 0.5, float, "magnitude of the perturbations of chi"),
        Parameter("gene_dup_rate", 0.0, float, "rate with which a gene is duplicated"),
        Parameter("gene_loss_rate", 0.0, float, "rate with which a gene is lost"),
        Parameter("num_comp_max", 32, int, "maximal number of components"),
    ]

    def __init__(
        self,
        parameters: dict[str, Any] = None,
        *,
        free_energy: FloryHuggins = None,
        rng=None,
    ):
        """
        Args:
            parameters (dict): Parameters of this individual
            free_energy (:class:`FloryHuggins`): Free energy governing this individual
            rng: the random number generator
        """
        super().__init__(parameters=parameters, rng=rng)

        # obtain the number of initial components
        num_comps = self.parameters["num_comp_init"]
        if num_comps in {None, NoValue}:
            # choose a default value for the number of components
            self.parameters["num_comp_init"] = 5

        # set some default parameters if they are not given
        if self.parameters["num_phases"] in {None, NoValue}:
            self.parameters["num_phases"] = self.parameters["num_comp_init"] + 2
        if self.parameters["chi_mean"] in {None, NoValue}:
            # this estimate originated from a numerical fit
            self.parameters["chi_mean"] = 3.0 + 0.3 * self.parameters["num_comp_init"]
        if self.parameters["chi_std"] in {None, NoValue}:
            self.parameters["chi_std"] = 1.0

        # initialize super class (this parse parameters again
        if free_energy is None:
            self.free_energy = self._get_random_free_energy()
        else:
            self.free_energy = free_energy
        if self.free_energy.inert_solvent != self.parameters["inert_solvent"]:
            self._logger.warning(
                "Free energy does not have `inert_solvent == "
                f"{self.parameters['inert_solvent']}`"
            )

        self._limit_interactions()

        self.diagnostics["chi_mean_value"] = self.parameters["chi_mean"]
        self.diagnostics["chi_std_value"] = self.parameters["chi_std"]

    def copy(self) -> FullChiIndividual:
        """create a copy of this individual"""
        return self.__class__(self.parameters, free_energy=self.free_energy.copy())

    def _limit_interactions(self) -> None:
        """ensures that the interactions stay within the given bounds"""
        chi_limit = self.parameters["chi_limit"]
        if chi_limit == "clip":
            # clip entries to certain range
            chi_min = self.parameters["chi_min"]
            chi_max = self.parameters["chi_max"]
            np.clip(self.free_energy.chis, chi_min, chi_max, out=self.free_energy.chis)

        elif chi_limit == "norm":
            # limit the mean interaction strength to a certain value
            norm_max = self.parameters["chi_norm_max"]
            values = self.free_energy.independent_entries
            norm = np.mean(np.abs(values))  # 1-norm
            if norm > norm_max:
                # rescale entries to obey limit
                self.free_energy.chis *= norm_max / norm

        elif chi_limit != "none":
            raise ValueError(f"Unknown limiting function `{chi_limit}`")

    def _get_random_free_energy(self) -> FloryHuggins:
        """determine a random free energy for this individual"""
        if self.parameters["num_comp_init"] in {None, NoValue}:
            raise ValueError("Component count was not specified")

        # choose a random mean chi
        if self.parameters["randomize_chi_mean"]:
            chi = np.clip(self.rng.normal(self.parameters["chi_mean"], 3), 0, 10)
        else:
            chi = self.parameters["chi_mean"]

        # choose an according random free energy
        return FloryHuggins.from_random_normal(
            num_comp=self.parameters["num_comp_init"],
            chi_mean=chi,
            chi_std=self.parameters["chi_std"],
            inert_solvent=self.parameters["inert_solvent"],
            rng=self.rng,
        )

    def randomize(self) -> None:
        """set random state of the individual"""
        self.free_energy = self._get_random_free_energy()
        self._limit_interactions()

    def cross(self, other: FullChiIndividual) -> FullChiIndividual:
        """create a cross between this and another individual"""
        f = self.free_energy.copy()
        num_comps = min(self.num_comps, other.num_comps)
        change_rows = np.random.random(num_comps) < 0.5
        for i in np.flatnonzero(change_rows):
            chi_replace = other.free_energy.chis[i, :num_comps]
            f.chis[:num_comps, i] = f.chis[i, :num_comps] = chi_replace

        # FIXME: This crossing is not very consistent since it does not faithfully
        # combine information from the two individuals on equal footing. In particular,
        # we do not even have information about individual components but instead
        # attempt to combine information the interactions of pairs of components.
        return self.__class__(parameters=self.parameters, free_energy=f)

    def _mutate_interaction_strengths(self):
        """mutate the entries of the interaction matrix"""
        # read parameters
        num_comp = self.free_energy.num_comps
        strategy = self.parameters["mutation_strategy"]
        mutation_size = self.parameters["mutation_size"]

        def change_one_entry(comp):
            """helper function changing a single random entry for a given component"""
            comp2 = np.random.randint(num_comp - 1)
            if comp2 >= comp:
                comp2 += 1  # ensure that diagonal is not modified

            Δchi = np.random.normal(0, mutation_size)
            self.free_energy.chis[comp, comp2] += Δchi
            self.free_energy.chis[comp2, comp] += Δchi

        if strategy == "single_entry":
            # change exactly one entry of the interaction matrix
            change_one_entry(np.random.randint(num_comp))

        elif strategy == "one_per_component":
            # change one entry for each component
            for comp in range(num_comp):
                change_one_entry(comp)

        elif strategy == "all_entries":
            # change all entries of the interaction matrix
            Δchi = np.zeros((num_comp, num_comp))
            Δchi[np.triu_indices_from(Δchi, 1)] = np.random.normal(
                0, mutation_size, size=num_comp * (num_comp - 1) // 2
            )
            self.free_energy.chis += Δchi + Δchi.T

        else:
            raise ValueError(f"Unknown mutation strategy '{strategy}'")

        # ensure that the interactions stay within their bounds
        self._limit_interactions()

    def _mutate_component_count(self):
        """duplicate or remove components"""
        # read parameters
        gene_dup_rate = self.parameters["gene_dup_rate"]
        gene_loss_rate = self.parameters["gene_loss_rate"]

        if gene_dup_rate + gene_loss_rate > 0:
            # duplicate or remove components
            num_comp = self.free_energy.num_comps

            delete_comps = []

            # iterate over all components and either duplicate them or register them
            # for later removal. We cannot remove them immediately, since this would
            # mess up the indexing
            for comp in range(num_comp):
                event_num = np.random.random()  # random number to decide what to do
                if event_num < gene_dup_rate:
                    # duplicate this component if there are not too many
                    num_comp_now = self.free_energy.num_comps - len(delete_comps)
                    if num_comp_now < self.parameters["num_comp_max"]:
                        self.free_energy = self.free_energy.duplicate_component(comp)

                elif event_num < gene_loss_rate + gene_dup_rate:
                    # register this component for removal
                    delete_comps.append(comp)

            # remove all components that are registered for removal
            if delete_comps:
                for comp in delete_comps[::-1]:  # reverse iteration to keep indices
                    if self.free_energy.num_comps > 1:
                        self.free_energy = self.free_energy.remove_component(comp)

    def mutate(self):
        """mutate the individual in place"""
        self._mutate_interaction_strengths()
        self._mutate_component_count()
        assert self.free_energy.is_consistent
