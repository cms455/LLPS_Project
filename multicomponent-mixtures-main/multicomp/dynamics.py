"""
Module defining the relaxation dynamics of mixtures.

There are two different versions available here, depending on whether the extensive
phase volumes are also described or not. The class :class:`RelaxationDynamics` describes
the relaxation of a :class:`MultiphaseSystem`, which only included the intensive volume
fractions of each phase. Conversely, :class:`ConsistentRelaxationDynamics` describes the
relaxation of :class:`MultiphaseVolumeSystem`, which additionally tracks phase volumes.

.. autosummary::
   :nosignatures:

   ~RelaxationDynamics
   ~ConsistentRelaxationDynamics

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from matplotlib import cm
from tqdm.auto import tqdm

from modelrunner.parameters import Parameter, Parameterized

from .mixture import MultiphaseSystem, MultiphaseVolumeSystem


class RelaxationDynamics(Parameterized):
    r"""class for handling the relaxation toward equilibrium

    We here implement the simplified relaxation dynamics given by the following system
    of ordinary differential equations:

    .. math::
        \partial_t \phi^{(n)}_i =
        \phi^{(n)}_i \sum_{m=1}^M  \left[
            \phi^{(m)}_i \bigl(\mu^{(m)}_i - \mu^{(n)}_i\bigr)
            + P^{(m)} - P^{(n)}
        \right]

    where :math:`\phi^{(n)}_i`, :math:`\mu^{(n)}_i`, and :math:`P^{(n)}` respectively
    denote volume fraction, chemical potential, and pressure of component :math:`i` in
    phase :math:`n`.
    """

    parameters_default = [
        Parameter(
            "convergence_criterion",
            "composition_change",
            str,
            "Determines how the relaxation concludes that the stationary state has "
            "been reached.",
        ),
        Parameter(
            "method",
            "explicit",
            str,
            "Selects the simulation method, which currently can be `explicit`, "
            "`explicit_adaptive`, or `implicit`.",
        ),
        Parameter(
            "adaptive_maxerror",
            1e-6,
            float,
            "Maximal error for the adaptive time evolution. This value cannot be "
            "changed after the first adaotive evolution of any RelaxationDynamics has "
            "been started.",
        ),
        Parameter(
            "implicit_maxiter",
            100,
            int,
            "Maximal implicit iterations. This value cannot be changed after the first "
            "implicit evolution of any RelaxationDynamics has been started.",
        ),
        Parameter(
            "implicit_maxerror",
            1e-3,
            float,
            "Maximal implicit error. This value cannot be changed after the first "
            "implicit evolution of any RelaxationDynamics has been started.",
        ),
        Parameter(
            "equilibrium_tolerance",
            np.inf,
            float,
            "The tolerance used to determine whether a stationary state is actually at"
            "equilibrium. Check is disabled for the default value `np.inf`.",
        ),
    ]

    _cache: dict[str, Any] = {}  # class level cache for functions

    dt_minimal = 1e-8
    """float: minimal allowed time step. An error is raised if lower dts are required"""
    dt_maximal = 1e4
    """float: maximal allowed time step"""

    def __init__(self, mixture: MultiphaseSystem, parameters: dict[str, Any] = None):
        """
        Args:
            mixture (:class:`MultiphaseSystem`): The mixture that we evolve
            parameters (dict): Parameters of this individual
        """
        super().__init__(parameters=parameters)
        self.mixture = mixture
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def num_comps(self):
        """the number of components in the simulation"""
        return self.mixture.num_comps

    @property
    def num_phases(self):
        """the number of phases in the simulation"""
        return self.mixture.num_phases

    def evolution_rate(self) -> np.ndarray:
        """the evolution rate of the dynamical system"""
        phis = self.mixture.phis
        mus = self.mixture.chemical_potentials
        ps = self.mixture.pressures

        # diffusive fluxes
        dN_1 = (phis * mus).sum(axis=0, keepdims=True)
        dN_2 = phis.sum(axis=0, keepdims=True) * mus
        dN = dN_1 - dN_2

        # accumulated pressure differences
        dp = self.num_phases * ps - ps.sum(axis=0, keepdims=True)

        return phis * (dN - dp[:, np.newaxis])  # type: ignore

    def _make_evolution_rate(self) -> Callable:
        """create a compiled method evaluating the evolution rate"""
        if "evolution_rate" not in self._cache:
            # create the evolution rate function and store it on a class-level cache
            calc_diffs = self.mixture.free_energy.make_calc_diffs()

            @nb.njit
            def evolution_rate(phis: np.ndarray, chis: np.ndarray = None) -> np.ndarray:
                """calculates the evolution rate of a system with given interactions"""
                num_phases, num_comps = phis.shape

                # get chemical potential and pressure for all components and phases
                mus = np.empty((num_phases, num_comps))
                ps = np.empty(num_phases)
                for n in range(num_phases):  # iterate over phases
                    mu, p = calc_diffs(phis[n], chis)
                    mus[n, :] = mu
                    ps[n] = p

                # calculate rate of change of the composition in all phases
                dc = np.zeros((num_phases, num_comps))
                for n in range(num_phases):
                    for m in range(num_phases):
                        delta_p = ps[n] - ps[m]
                        for i in range(num_comps):
                            delta_mu = mus[m, i] - mus[n, i]
                            dc[n, i] += phis[n, i] * (phis[m, i] * delta_mu - delta_p)
                return dc

            self._cache["evolution_rate"] = evolution_rate

        return self._cache["evolution_rate"]  # type: ignore

    def _make_evolver_explicit(self) -> Callable:
        """create a compiled method iterating the dynamics forward in time explicitly"""
        if "evolver_explicit" not in self._cache:
            # create the iterator and store it on a class-level cache

            rhs = self._make_evolution_rate()

            @nb.njit
            def iterate_inner(
                phis: np.ndarray,
                t: float,
                t_next: float,
                dt: float,
                chis: np.ndarray = None,
            ) -> tuple[int, float, float]:
                """iterates a system with given interactions"""
                steps = 0
                while t < t_next:
                    # make a step
                    phis += dt * rhs(phis, chis)
                    t += dt
                    steps += 1

                    # check validity of the result
                    if np.any(np.isnan(phis)):
                        raise RuntimeError("Encountered NaN")
                    elif np.any(phis <= 0):
                        raise RuntimeError("Non-positive concentrations")
                    elif np.any(phis.sum(axis=-1) >= 1):
                        raise RuntimeError("Non-positive solvent concentrations")

                return steps, t, dt

            self._cache["evolver_explicit"] = iterate_inner

        return self._cache["evolver_explicit"]  # type: ignore

    def _make_evolver_explicit_adaptive(self) -> Callable:
        """create a compiled method iterating the dynamics forward in time explicitly"""
        if "evolver_explicit_adpative" not in self._cache:
            # create the iterator and store it on a class-level cache

            rhs = self._make_evolution_rate()
            error_tolerance = self.parameters["adaptive_maxerror"]
            dt_min = self.dt_minimal
            dt_max = self.dt_maximal
            dt_min_nan_err = f"Encountered NaN even though dt < {dt_min}"
            dt_min_err = f"Time step below {dt_min}"

            @nb.njit
            def iterate_inner(
                phis: np.ndarray,
                t: float,
                t_next: float,
                dt: float,
                chis: np.ndarray = None,
            ) -> tuple[int, float, float]:
                """iterates a system with given interactions"""
                steps = 0
                dt_opt = dt
                rate = rhs(phis, chis)  # calculate initial rate

                while t < t_next:
                    # use a smaller (but not too small) time step if close to t_next
                    dt_step = max(min(dt_opt, t_next - t), dt_min)

                    # single step with dt
                    step_large = phis + dt_step * rate

                    # double step with half the dt
                    step_small = phis + 0.5 * dt_step * rate

                    try:
                        # calculate rate at the midpoint of the double step
                        rate_midpoint = rhs(step_small, chis)
                    except Exception:
                        # an exception likely signals that rate could not be calculated
                        error_rel = np.nan
                    else:
                        # advance to end of double step
                        step_small += 0.5 * dt_step * rate_midpoint

                        # calculate maximal error
                        error = np.abs(step_large - step_small).max()
                        # normalize error to given tolerance
                        error_rel = error / error_tolerance

                        # check validity of the result
                        valid = (
                            np.all(np.isfinite(step_small))
                            and np.all(step_small > 0)
                            and np.all(step_small.sum(axis=-1) <= 1)
                        )
                        if not valid:
                            error_rel = np.nan

                    # do the step if the error is sufficiently small
                    if error_rel <= 1:
                        try:
                            # calculating the rate at putative new step
                            rate = rhs(step_small, chis)
                        except Exception:
                            # calculating the rate failed => retry with smaller dt
                            error_rel = np.nan
                        else:
                            # everything worked => do the step
                            steps += 1
                            t += dt_step
                            phis[...] = step_small

                    # adjust the time step
                    if error_rel < 0.00057665:
                        # error was very small => maximal increase in dt
                        # The constant on the right hand side of the comparison is chosen to
                        # agree with the equation for adjusting dt below
                        dt *= 4.0
                    elif np.isnan(error_rel):
                        # state contained NaN => decrease time step strongly
                        dt *= 0.25
                    else:
                        # otherwise, adjust time step according to error
                        dt *= max(0.9 * error_rel**-0.2, 0.1)

                    # limit time step to permissible bracket
                    if dt > dt_max:
                        dt = dt_max
                    elif dt < dt_min:
                        if np.isnan(error_rel):
                            raise RuntimeError(dt_min_nan_err)
                        else:
                            raise RuntimeError(dt_min_err)

                return steps, t, dt

            self._cache["evolver_explicit_adpative"] = iterate_inner

        return self._cache["evolver_explicit_adpative"]  # type: ignore

    def _make_evolver_implicit(self) -> Callable:
        """create a compiled method iterating the dynamics forward in time implicitly"""
        if "evolver_implicit" not in self._cache:
            # create the iterator and store it on a class-level cache

            rhs = self._make_evolution_rate()

            # parameters of the implicit algorithm
            maxiter = self.parameters["implicit_maxiter"]
            maxerror = self.parameters["implicit_maxerror"]

            @nb.njit
            def iterate_inner(
                phis: np.ndarray,
                t: float,
                t_next: float,
                dt: float,
                chis: np.ndarray = None,
            ) -> tuple[int, float, float]:
                """iterates a system with given interactions"""
                steps = 0
                while t < t_next:
                    # make a step
                    rate_last = dt * rhs(phis, chis)  # initialize guess for the rate
                    t += dt
                    steps += 1

                    for _ in range(maxiter):
                        phis_guess = phis + rate_last

                        # check validity of the result
                        if np.any(np.isnan(phis_guess)):
                            raise RuntimeError("Encountered NaN")
                        elif np.any(phis_guess <= 0):
                            raise RuntimeError("Non-positive concentrations")
                        elif np.any(phis_guess.sum(axis=-1) >= 1):
                            raise RuntimeError("Non-positive solvent concentrations")

                        rate_this = dt * rhs(phis_guess, chis)

                        # calculate the error
                        err = phis_guess - phis - rate_this
                        if np.linalg.norm(err) < maxerror:
                            break  # implicit step converged

                        rate_last = rate_this

                    else:
                        raise RuntimeError("Implicit step did not converge")

                    phis += rate_this

                return steps, t, dt

            self._cache["evolver_implicit"] = iterate_inner

        return self._cache["evolver_implicit"]  # type: ignore

    def evolve(
        self,
        t_range: float,
        dt: float = 1.0,
        *,
        interval: float = 1.0,
        tolerance: float = 1e-5,
        progress: bool = True,
        save_intermediate_data: bool = False,
    ):
        """use explicit Euler method with an adaptive time step to evolve ODEs

        Args:
            t_range:
                The time for which the simulation is maximally run. It might abort
                earlier when the stationary state is reached.
            dt:
                The initial time step. This time step will be automatically reduced when
                the simulation has trouble advancing.
            interval:
                The time interval at which convergence is checked (and data is saved if
                `save_intermediate_data` is True.
            tolerance:
                The tolerance determining when the simulation will be aborted because
                the algorithm thinks we reached the stationary state
            progress:
                Flag determining whether to show a progress bar during the simulation.
            save_intermediate_data:
                Flag determining whether saving (and returning) data in time intervals
                determined by `interval`

        Returns:
            tuple: Simulation time and mixture. If `save_intermediate_data == True`,
            these are lists with all the saved data. Otherwise, only the last time point
            is returned.
        """
        # steps_tracker = int(np.ceil(t_range / interval))
        tracker_times = np.r_[np.arange(0, t_range, interval)[1:], t_range]
        # steps_inner = max(1, int(np.ceil(t_range / dt)) // steps_tracker)
        crit = self.parameters["convergence_criterion"]
        tol = tolerance * interval

        assert self.mixture.is_consistent
        if self.parameters["method"] == "explicit":
            iterate_inner = self._make_evolver_explicit()
        elif self.parameters["method"] == "explicit_adaptive":
            iterate_inner = self._make_evolver_explicit_adaptive()
        elif self.parameters["method"] == "implicit":
            iterate_inner = self._make_evolver_implicit()
        else:
            raise ValueError(f"Unknown relaxation method `{self.parameters['method']}`")

        t: float = 0
        num_steps = 0
        phis = self.mixture.phis.copy()  # do not change initial value
        sys = MultiphaseSystem(self.mixture.free_energy, phis)
        time_start = time.process_time()

        mixtures = []
        times: list[float] = []
        for t_next in tqdm(tracker_times, disable=not progress):
            # store result
            if save_intermediate_data:
                times.append(t)
                mixtures.append(self.mixture.copy(phis))
                phis_last = mixtures[-1].phis
            else:
                phis_last = phis.copy()

            # do the inner steps
            while True:
                try:
                    steps_inner, t_inner, dt = iterate_inner(
                        phis,
                        t=t,
                        t_next=t_next,
                        dt=dt,
                        chis=self.mixture.free_energy.chis,
                    )
                except RuntimeError as err:
                    # problems in the simulation => reduced dt and reset phis
                    dt /= 2
                    phis[:] = phis_last

                    self._logger.info(f"Reduced time step to {dt}")
                    if dt < self.dt_minimal:
                        raise RuntimeError(str(err) + "\nReached minimal time step.")
                else:
                    # step seemed to be ok
                    t = t_inner
                    num_steps += steps_inner
                    break

            # check distance and abort simulations if things do not change much
            if crit == "composition_change":
                if np.allclose(phis, phis_last, rtol=tol, atol=tol):
                    self._logger.info("Composition reached stationary state")
                    break

            elif crit == "entropy_production":
                if sys.entropy_production < tolerance:
                    self._logger.info("Entropy production reached tolerance")
                    break

            elif crit != "none":
                raise ValueError(f"Undefined convergence criterion: {crit}")

        # add final data point
        if not times or times[-1] != t:
            times.append(t)
            mixtures.append(self.mixture.copy(phis))

        # store diagnostic output
        self.diagnostics = {
            "runtime": time.process_time() - time_start,
            "num_steps": num_steps,
            "steps_tracker": len(tracker_times),
            "dt_last": dt,
        }

        # check whether equilibrium was actually reached
        if self.parameters["equilibrium_tolerance"] != np.inf:
            equi = self.mixture.at_equilibrium(self.parameters["equilibrium_tolerance"])
            self.diagnostics["at_equilibrium"] = equi

        if save_intermediate_data:
            return np.array(times), mixtures
        else:
            return times[0], mixtures[0]


class ConsistentRelaxationDynamics(Parameterized):
    r"""class for handling the relaxation toward equilibrium including volumes


    We here implement physical relaxation dynamics inspired by linear non-equilibrium
    thermodynamics, given by the following system of ordinary differential equations:

    .. math::
        \begin{align}
        \partial_t V^{(n)} &= \sum_{m=1}^M \left[P^{(n)} - P^{(m)}\right]
        \\
        \partial_t \phi^{(n)}_i &= 
        \frac{\phi^{(n)}_i}{V^{(n)}} \sum_{m=1}^M  \left[
            \phi^{(m)}_i \bigl(\mu^{(m)}_i - \mu^{(n)}_i\bigr)
            + P^{(m)} - P^{(n)}
        \right]
        \end{align}

    where :math:`\phi^{(n)}_i`, :math:`\mu^{(n)}_i`, and :math:`P^{(n)}` respectively
    denote volume fraction, chemical potential, and pressure of component :math:`i` in
    phase :math:`n`. Additionally, :math:`V^{(n)}` is the volume of the :math:`n`-th
    phase measured in terms of the molecular volumes, which are  assumed to be equal for
    all components for simplicity.
    """

    parameters_default = [
        Parameter(
            "convergence_criterion",
            "composition_change",
            str,
            "Determines how the relaxation concludes that the stationary state has "
            "been reached.",
        ),
    ]

    _cache: dict[str, Any] = {}  # class level cache for functions

    def __init__(
        self, mixture: MultiphaseVolumeSystem, parameters: dict[str, Any] = None
    ):
        """
        Args:
            mixture (:class:`MultiphaseSystem`): The mixture that we evolve
            parameters (dict): Parameters of this individual
        """
        super().__init__(parameters=parameters)
        self.mixture = mixture
        self._logger = logging.getLogger(self.__class__.__name__)

    @property
    def num_comps(self):
        """the number of components in the simulation"""
        return self.mixture.num_comps

    @property
    def num_phases(self):
        """the number of phases in the simulation"""
        return self.mixture.num_phases

    def evolution_rate(self) -> tuple[np.ndarray, np.ndarray]:
        """the evolution rate of the dynamical system"""
        phis = self.mixture.phis
        volumes = self.mixture.volumes
        mus = self.mixture.chemical_potentials
        ps = self.mixture.pressures

        dphis_dt = np.empty((self.num_phases, self.num_comps))
        dvols_dt = np.empty(self.num_phases)
        for n in range(self.num_phases):
            dvols_dt[n] = self.num_phases * ps[n] - ps.sum()
            for i in range(self.num_comps):
                dmu = phis[:, i] @ mus[:, i] - self.num_comps * mus[n, i]
                dphis_dt[n, i] = phis[n, i] / volumes[n] * (dmu - dvols_dt[n])

        return dphis_dt, dvols_dt

    def _make_evolution_rate(self) -> Callable:
        """create a compiled method evaluating the evolution rate"""
        if "evolution_rate" not in self._cache:
            # create the evolution rate function and store it on a class-level cache
            calc_diffs = self.mixture.free_energy.make_calc_diffs()

            @nb.njit
            def evolution_rate(
                phis: np.ndarray, volumes: np.ndarray, chis: np.ndarray = None
            ) -> tuple[np.ndarray, np.ndarray]:
                """calculates the evolution rate of a system with given interactions"""
                num_phases, num_comps = phis.shape

                # get chemical potential and pressure for all components and phases
                mus = np.empty((num_phases, num_comps))
                ps = np.empty(num_phases)
                for n in range(num_phases):  # iterate over phases
                    mu, p = calc_diffs(phis[n], chis)
                    mus[n, :] = mu
                    ps[n] = p

                # calculate rate of change of the composition in all phases
                dphis_dt = np.zeros((num_phases, num_comps))
                dvols_dt = np.zeros(num_phases)
                for n in range(num_phases):
                    dvols_dt[n] = num_phases * ps[n]
                    for m in range(num_phases):
                        dvols_dt[n] -= ps[m]
                        delta_p = ps[n] - ps[m]
                        for i in range(num_comps):
                            delta_mu = mus[m, i] - mus[n, i]
                            a = phis[m, i] * delta_mu - delta_p
                            dphis_dt[n, i] += phis[n, i] * a / volumes[n]
                return dphis_dt, dvols_dt

            self._cache["evolution_rate"] = evolution_rate

        return self._cache["evolution_rate"]  # type: ignore

    def _make_evolver_explicit(self) -> Callable:
        """create a compiled method iterating the dynamics forward in time explicitly"""
        if "evolver_explicit" not in self._cache:
            # create the iterator and store it on a class-level cache

            rhs = self._make_evolution_rate()

            @nb.njit
            def iterate_inner(
                phis: np.ndarray,
                volumes: np.ndarray,
                dt: float,
                steps: int,
                chis: np.ndarray = None,
            ) -> float:
                """iterates a system with given interactions"""
                for _ in range(steps):
                    # make a step
                    dphis, dvols = rhs(phis, volumes, chis)
                    phis += dt * dphis
                    volumes += dt * dvols

                    # check validity of the result
                    if np.any(np.isnan(phis)):
                        raise RuntimeError("Encountered NaN")
                    elif np.any(phis <= 0):
                        raise RuntimeError("Non-positive concentrations")
                    elif np.any(phis.sum(axis=-1) >= 1):
                        raise RuntimeError("Non-positive solvent concentrations")
                    elif np.any(volumes <= 0):
                        raise RuntimeError("Non-positive volume")

                return dt * steps

            self._cache["evolver_explicit"] = iterate_inner

        return self._cache["evolver_explicit"]  # type: ignore

    def evolve(
        self,
        t_range: float,
        dt: float = 1.0,
        *,
        interval: float = 1.0,
        tolerance: float = 1e-5,
        progress: bool = True,
        save_intermediate_data: bool = False,
    ):
        """use explicit Euler method with an adaptive time step to evolve ODEs

        Args:
            t_range:
                The time for which the simulation is maximally run. It might abort
                earlier when the stationary state is reached.
            dt:
                The initial time step. This time step will be automatically reduced when
                the simulation has trouble advancing.
            interval:
                The time interval at which convergence is checked (and data is saved if
                `save_intermediate_data` is True.
            tolerance:
                The tolerance determining when the simulation will be aborted because
                the algorithm thinks we reached the stationary state
            progress:
                Flag determining whether to show a progress bar during the simulation.
            save_intermediate_data:
                Flag determining whether saving (and returning) data in time intervals
                determined by `interval`

        Returns:
            tuple: Simulation time and mixture. If `save_intermediate_data == True`,
            these are lists with all the saved data. Otherwise, only the last time point
            is returned.
        """
        steps_tracker = int(np.ceil(t_range / interval))
        steps_inner = max(1, int(np.ceil(t_range / dt)) // steps_tracker)
        crit = self.parameters["convergence_criterion"]
        tol = tolerance * interval

        assert self.mixture.is_consistent
        iterate_inner = self._make_evolver_explicit()

        t = 0
        num_steps = 0
        phis = self.mixture.phis.copy()  # do not change initial value
        volumes = self.mixture.volumes.copy()
        sys = MultiphaseVolumeSystem(self.mixture.free_energy, phis, volumes)

        mixtures = []
        times = []
        for _ in tqdm(range(steps_tracker), disable=not progress):
            # store result
            if save_intermediate_data:
                times.append(t)
                mixtures.append(self.mixture.copy(phis, volumes))
                phis_last = mixtures[-1].phis
                volumes_last = mixtures[-1].volumes
            else:
                phis_last = phis.copy()
                volumes_last = volumes.copy()

            # do the inner steps
            while True:
                try:
                    t_inner = iterate_inner(
                        phis,
                        volumes,
                        dt,
                        steps=steps_inner,
                        chis=self.mixture.free_energy.chis,
                    )
                except RuntimeError as err:
                    # problems in the simulation => reduced dt and reset phis
                    dt /= 2
                    steps_inner *= 2
                    phis[:] = phis_last
                    volumes[:] = volumes_last

                    self._logger.info(f"Reduced time step to {dt}")
                    if dt < 1e-7:
                        raise RuntimeError(str(err) + "\nReached minimal time step.")
                else:
                    # step seemed to be ok
                    t += t_inner
                    num_steps += steps_inner
                    break

            # check distance and abort simulations if things do not change much
            if crit == "composition_change":
                phis_close = np.allclose(phis, phis_last, rtol=tol, atol=tol)
                vols_close = np.allclose(volumes, volumes_last, rtol=tol, atol=tol)
                if phis_close and vols_close:
                    self._logger.info("Composition and sizes reached stationary state")
                    break

            elif crit == "entropy_production":
                if sys.entropy_production < tolerance:
                    self._logger.info("Entropy production reached tolerance")
                    break

            elif crit != "none":
                raise ValueError(f"Undefined convergence criterion: {crit}")

        # add final data point
        if not times or times[-1] != t:
            times.append(t)
            mixtures.append(self.mixture.copy(phis, volumes))

        # store diagnostic output
        self.diagnostics = {
            "num_steps": num_steps,
            "steps_tracker": steps_tracker,
            "dt_last": dt,
        }

        if save_intermediate_data:
            return np.array(times), mixtures
        else:
            return times[0], mixtures[0]


def plot_convergence(
    times,
    mixtures,
    incl_ptp: bool = True,
    incl_entropy: bool = True,
    incl_volumes: bool = False,
):
    """plot the results of a simulation to test for convergence

    Args:
        times: The time points where data was saved
        mixtures: Information about the mixtures for each time point
        incl_ptp (bool): Whether to also plot the peak-to-peak values
        incl_entropy (bool): Whether to also plot the entropy production rate
        incl_volumes (bool): Whether to also show volume evolution
    """
    data = np.empty((len(mixtures), 5))
    for i, m in enumerate(mixtures):
        ps = m.pressures
        mus = m.chemical_potentials

        # variation in pressure
        data[i, 0] = ps.std()  # standard deviation (STD)
        data[i, 1] = ps.ptp()  # peak-to-peak (PTP)

        # variation in chemical potential
        data[i, 2] = mus.std(axis=0).mean()  # mean CV
        data[i, 3] = mus.ptp(axis=0).max()  # max PTP

        # rate of free energy decrease
        dmu = (mus[:, np.newaxis, :] - mus[np.newaxis, :, :]) ** 2
        dp = (ps[:, np.newaxis] - ps[np.newaxis, :]) ** 2
        data[i, 4] = np.einsum("ni,mi,nmi->", m.phis, m.phis, dmu) + dp.sum()

    plt.plot(times, data[:, 0], "-C0", label="Pressure STD")
    if incl_ptp:
        plt.plot(times, data[:, 1], ":C0", label="Pressure PTP")
    plt.plot(times, data[:, 2], "-C1", label="Chemical potential mean STD")
    if incl_ptp:
        plt.plot(times, data[:, 3], ":C1", label="Chemical potential max. PTP")
    if incl_entropy:
        plt.plot(times, data[:, 4], "-C2", label="Entropy production rate")
    if incl_volumes:
        plt.plot(times, [m.volumes.std() for m in mixtures], "-C3", label="Volumes STD")

    plt.xlabel("Time")
    plt.ylabel("Average deviation")

    plt.legend(loc="best")
    plt.yscale("log")


def plot_concentrations(
    times,
    mixtures,
    display="norm_last",
    cmap=cm.viridis,
    color_by_cluster: bool = False,
    **kwargs,
):
    """plot the results of a simulation to show evolution of the concentrations

    Args:
        times:
            The time points where data was saved
        mixtures:
            Information about the mixtures for each time point
        display:
            Determines what to display. Options include:
            * `norm_last`: The dot product with the last time point
            * `norm_first`: The dot product with the first time point
            * `min`, `mean`, `max`: Show one component per phase
            * `std`: Standard deviation of concentrations per phase
        cmap:
            Colormap for assigning colors
        color_by_cluster:
            Whether all trajectories ending in the same cluster have the same color
        **kwargs:
            All additional arguments are forwarded to the plotting function
    """
    phis = np.array([m.phis for m in mixtures])

    if display == "norm_last":
        data = np.einsum("tni,ni->tn", phis, phis[-1])
        title = r"Norm $\sum_i c^{(n)}_i(t) \cdot c^{(n)}_i(t_\mathrm{last})$ for all phases $n$"
    elif display == "norm_first":
        data = np.einsum("tni,ni->tn", phis, phis[0])
        title = r"Norm $\sum_i c^{(n)}_i(t) \cdot c^{(n)}_i(t_0)$ for all phases $n$"
    elif display == "min":
        data = phis.min(axis=-1)
        title = "Minimal concentration per phase"
    elif display == "max":
        data = phis.max(axis=-1)
        title = "Maximal concentration per phase"
    elif display == "mean":
        data = phis.mean(axis=-1)
        title = "Mean concentration per phase"
    elif display == "std":
        data = phis.std(axis=-1)
        title = "STD of concentration per phase"
    elif display in {"var", "variance"}:
        data = phis.var(axis=-1)
        title = r"Variance of concentration per phase"
    else:
        data = phis[:, :, display]
        title = f"Concentration of species {display}"

    if color_by_cluster:
        color_id = mixtures[-1]._get_clusters()
        num_clusters = max(color_id)
        labeled = [False, False]
    else:
        color_id = range(mixtures[-1].num_phases)

    num_phases = mixtures[0].num_phases
    for n in range(num_phases):
        color = cmap(color_id[n] / max(color_id))
        if color_by_cluster:
            if color_id[n] == 1 and not labeled[0]:
                label = f"Cluster 1"
                labeled[0] = True
            elif color_id[n] == num_clusters and not labeled[1]:
                label = f"Cluster {num_clusters}"
                labeled[1] = True
            else:
                label = ""
        else:
            label = f"Phase {n}" if n == 0 or n == num_phases - 1 else ""
        plt.plot(times, data[:, n], color=color, label=label, **kwargs)

    plt.xlabel("Time")
    plt.legend(loc="best")
    plt.title(title)
