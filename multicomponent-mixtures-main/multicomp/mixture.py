"""
Defines classes that contain the state of a multiphase system.

All classes describe a system of :math:`M` phases that are filled with a multicomponent
mixture of :math:`N` components and a solvent. The bare minimum of information to
describe this state is captured by :class:`Mixture`, which just contains the fractions
of all components in all phases. The class :class:`MultiphaseSystem` additionally
contains information about the free energy that described the interactions of the
components. Finally, the class :class:`MultiphaseVolumeSystem` also adds information
about the volume of the individual phases. 

.. autosummary::
   :nosignatures:

   ~Mixture
   ~MultiphaseSystem
   ~MultiphaseVolumeSystem

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import warnings

import numpy as np
from numba import njit
from scipy import cluster, spatial

from .thermodynamics import FloryHuggins


def get_composition_angles(phis: np.ndarray, *, squareform: bool = False) -> np.ndarray:
    r"""calculate composition angles between phases

    Args:
        phis (:class:`numpy.ndarray`): The composition of all phases
        squareform: Whether the square matrix or only distinct entries are returned

    Returns:
        A vector or a matrix containing the angles in the interval :math:`[0, \pi]`
    """
    # calculate all distinct angles
    dist = spatial.distance.pdist(phis, metric="cosine")
    angles = np.arccos(1 - dist)
    # return result in correct form
    return spatial.distance.squareform(angles) if squareform else angles  # type: ignore


@njit
def get_uniform_random_composition_nb(num_comps: int) -> np.ndarray:
    """pick concentrations uniform from allowed simplex (sum of fractions < 1)

    Args:
        num_comps (int): the number of components to use

    Returns:
        An array with `num_comps` random fractions
    """
    phis = np.empty(num_comps)
    phi_max = 1.0
    for d in range(num_comps):
        x = np.random.beta(1, num_comps - d) * phi_max
        phi_max -= x
        phis[d] = x
    return phis


def get_uniform_random_composition_np(num_comps: int, rng=None) -> np.ndarray:
    """pick concentrations uniform from allowed simplex (sum of fractions < 1)

    Args:
        num_comps (int): the number of components to use
        rng: The random number generator

    Returns:
        An array with `num_comps` random fractions
    """
    if rng is None:
        rng = np.random.default_rng()

    phis = np.empty(num_comps)
    phi_max = 1.0
    for d in range(num_comps):
        x = rng.beta(1, num_comps - d) * phi_max
        phi_max -= x
        phis[d] = x
    return phis


class Mixture:
    """represents a mixture of multiple phases

    Attributes:
        phi (:class:`~numpy.ndarray`):
            The fraction of all components in all phases. The first axis enumerates the
            phases while the second axis enumerates components
    """

    def __init__(self, phis: np.ndarray):
        """
        Args:
            phis: The composition vectors for each phase (a 2d array)
        """
        self.phis = np.asarray(phis)

    @property
    def num_phases(self) -> int:
        """the number of phases in the system"""
        return self.phis.shape[0]

    @property
    def num_comps(self) -> int:
        """the number of components (that are not the solvent)"""
        return self.phis.shape[1]

    @property
    def solvent_concentration(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Fraction occupied by the solvent"""
        return 1 - self.phis.sum(axis=1)  # type: ignore

    @property
    def phi_all(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Fraction of all species including the solvent"""
        return np.c_[self.phis, self.solvent_concentration]  # type: ignore

    @property
    def phi_bar(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Mean fraction averaged over phases"""
        return self.phis.mean(axis=0)  # type: ignore

    @property
    def enrichment(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Enrichment of each component in each phase"""
        return self.phis / self.phi_bar  # type: ignore

    @classmethod
    def from_random_composition(
        cls,
        num_phases: int,
        num_comps: int,
        dist: str = "simplex_uniform",
        *,
        rng=None,
        **kwargs,
    ) -> Mixture:
        """create multiphase system with random compositions

        Args:
            num_phases: The number of phases that should be created
            num_comps: The number of different components that should be build
            dist: Determines the distribution determining the weights of the components
            rng: The random number generator
        """
        obj = cls(np.empty((num_phases, num_comps)))
        obj.set_random_composition(dist=dist, rng=rng, **kwargs)
        return obj

    def set_random_composition(
        self,
        dist: str = "simplex_uniform",
        *,
        num_phases: int = None,
        rng=None,
        **kwargs,
    ):
        """choose random compositions for the phases

        Args:
            dist: Determines the distribution determining the weights of the components
            num_phases: The number of phases. If omitted, the number is not changed.
            rng: The random number generator
        """
        if num_phases is None:
            num_phases = self.num_phases
        if num_phases != self.num_phases:
            self.phis = np.empty((num_phases, self.num_comps))

        if rng is None:
            rng = np.random.default_rng()

        if dist == "simplex_uniform" or dist == "uniform_simplex":
            # special distribution that picks non-solvent components uniformly in the
            # allowed range, i.e., so that the sum is in [0, 1]
            for i in range(self.num_phases):
                self.phis[i] = get_uniform_random_composition_np(
                    self.num_comps, rng=rng
                )

        elif dist == "simplex_neighborhood":
            # special distribution that picks an average fraction of all components
            # uniformly in the allowed range, i.e., so that the sum is in [0, 1], and
            # then chooses components in a suitable neighborhood
            phibars = get_uniform_random_composition_np(self.num_comps, rng=rng)

            delta = 0.99 * np.min(phibars)  # distance to axes of phase diagram
            # we could remove the `np.min` to have a rectangular search area
            for i in range(self.num_phases):
                # pick random composition in the neighboorhood of phibar.
                # Our choice of delta ensures that self.phis[i] is positive!
                self.phis[i] = phibars + rng.uniform(-delta, delta, self.num_comps)
                # Check whether the solvent fraction is also positive
                phis_sum = self.phis[i].sum()
                if phis_sum > 0.99:
                    # if not, correct composition, so the solvent fraction is positive
                    self.phis[i] *= 0.99 / phis_sum

        elif dist == "simplex_neighborhood_ring":
            # special distribution that picks an average fraction of all components
            # uniformly in the allowed range, i.e., so that the sum is in [0, 1], and
            # then chooses components in a suitable annulus-like neighborhood with a
            # maximal radius
            phibars = get_uniform_random_composition_np(self.num_comps, rng=rng)

            delta = 0.99 * np.min(phibars)  # distance to axes of phase diagram
            for i in range(self.num_phases):
                # pick random composition in the neighboorhood of phibar.
                # Our choice of delta ensures that self.phis[i] is positive!
                direction = rng.normal(0, 1, self.num_comps)
                direction /= np.linalg.norm(direction)
                self.phis[i] = phibars + delta * direction
                # Check whether the solvent fraction is also positive
                phis_sum = self.phis[i].sum()
                if phis_sum > 0.99:
                    # if not, correct composition, so the solvent fraction is positive
                    self.phis[i] *= 0.99 / phis_sum

        else:
            # draw uniform weights for all N + 1 components
            size = (num_phases, self.num_comps + 1)
            if dist == "uniform":
                weights = rng.uniform(0, 1, size=size)
            elif dist == "lognormal":
                weights = rng.lognormal(size=size, **kwargs)
            else:
                raise ValueError(f"Unknown dist `{dist}`")

            # determine the volume fractions of the first N components
            total_weights = np.sum(weights, axis=1, keepdims=True)
            self.phis[:] = weights[:, : self.num_comps] / total_weights

    def _get_clusters(self, dist: float = 1e-2):
        """determine the distinct phases in the composition

        Args:
            dist (float): Cut-off distance for cluster analysis
        """
        # calculate distances between compositions
        dists = spatial.distance.pdist(self.phis)
        # obtain hierarchy structure
        links = cluster.hierarchy.linkage(dists, method="centroid")
        # flatten the hierarchy by clustering
        return cluster.hierarchy.fcluster(links, dist, criterion="distance")

    def get_clusters(self, dist: float = 1e-2) -> np.ndarray:
        """return the concentrations in the distinct clusters

        Args:
            dist (float): Cut-off distance for cluster analysis
        """
        clusters = self._get_clusters(dist)
        return np.array(
            [self.phis[clusters == n, :].mean(axis=0) for n in np.unique(clusters)]
        )

    def count_clusters(self, dist: float = 1e-2) -> int:
        """calculate the number of distinct phases

        Args:
            dist (float): Cut-off distance for cluster analysis
        """
        return int(self._get_clusters(dist).max())

    def enriched_components(self, threshold: float = 1.2) -> np.ndarray:
        """return which components are enriched in which phase

        Args:
            threshold (float):
                Determines how much larger the fraction needs to be compared to the
                average fraction to be counted as enriched.

        Returns:
            :class:`numpy.ndarray`: A binary matrix denoting enrichments
        """
        if threshold < 1:
            warnings.warn("Enrichment requires threshold > 1")
        return self.enrichment > threshold

    def count_enriched_components(self, threshold: float = 1.2) -> np.ndarray:
        """return how many components are enriched per phase

        Args:
            threshold (float):
                Determines how much larger the fraction needs to be compared to the
                average fraction to be counted as enriched.

        Returns:
            :class:`numpy.ndarray`: The number of enriched components per phase
        """
        return self.enriched_components(threshold).sum(axis=1)  # type: ignore

    def composition_angles(
        self,
        cluster: bool = True,
        *,
        dist: float = 1e-2,
        squareform: bool = False,
    ) -> np.ndarray:
        r"""calculate composition angles between phases

        Args:
            cluster: Whether clusters or all phases are included
            dist (float): Cut-off distance for cluster analysis
            squareform: Whether the square matrix or only distinct entries are returned

        Returns:
            A vector or a matrix containing the angles in the interval :math:`[0, \pi]`
        """
        # get the composition of the phases to consider
        phis = self.get_clusters(dist) if cluster else self.phis
        # return associated mixing angles
        return get_composition_angles(phis, squareform=squareform)


class MultiphaseSystem(Mixture):
    """represents a multicomponent system of multiple phases"""

    def __init__(self, free_energy: FloryHuggins, phis: np.ndarray):
        """
        Args:
            free_energy: The multicomponent free energy
            phis: The composition vectors for each phase (a 2d array)
        """
        super().__init__(phis)
        self.free_energy = free_energy
        assert self.is_consistent

    @classmethod
    def from_demixed_composition(
        cls, free_energy: FloryHuggins, c_dense: float = 0.5
    ) -> MultiphaseSystem:
        """create multiphase system with a demixed composition

        This will create one more phase than there are components

        Args:
            free_energy: The multicomponent free energy
            c_dense: The concentration of the dense phases
        """
        N = free_energy.num_comps
        c0 = (1 - c_dense) / N  # dilute concentrations
        phis = np.full((N + 1, N), c0)
        phis[np.diag_indices(N)] = c_dense
        return cls(free_energy, phis)

    @classmethod
    def from_unstable_composition(
        cls, free_energy: FloryHuggins, samples: int = 100
    ) -> MultiphaseSystem:
        """create multiphase system using an unstable mode

        A RuntimeError is raised if such a mode cannot be found. This will inevitably be
        the case if the free energy is outside the spinodal region!

        Args:
            samples: The number of compositions that are tried to find an unstable mode
        """
        m = cls.from_random_composition(free_energy, num_phases=samples)

        # find an unstable composition
        for phi in m.phis:
            if not free_energy.is_stable(phi):
                break
        else:
            raise RuntimeError("Could not find an unstable mode")

        # find the unstable mode
        vals, vecs = np.linalg.eig(free_energy.hessian(phi))
        assert np.any(vals < 0)  # there must be a negative eigenvalue
        unstable_mode = vecs[np.argmin(vals)]

        # create a composition using the unstable mode
        phis = phi[np.newaxis, :] + 1e-2 * np.diag(unstable_mode)
        return cls(free_energy, phis)

    @classmethod
    def from_random_composition(  # type: ignore
        cls,
        free_energy: FloryHuggins,
        num_phases: int,
        dist: str = "simplex_uniform",
        *,
        rng=None,
        **kwargs,
    ) -> MultiphaseSystem:
        """create multiphase system with random compositions

        Args:
            free_energy: The multicomponent free energy
            num_phases: The number of phases that should be created
            dist: Determines the distribution determining the weights of the components
            rng: The random number generator
        """
        obj = cls(free_energy, np.zeros((num_phases, free_energy.num_comps)))
        obj.set_random_composition(dist=dist, rng=rng, **kwargs)
        return obj

    def copy(self, phis: np.ndarray = None) -> MultiphaseSystem:
        """copy the system (with potentially a new composition)"""
        if phis is None:
            phis = self.phis
        return self.__class__(self.free_energy, phis.copy())

    @property
    def is_consistent(self) -> bool:
        """does some self-consistency checks"""
        return self.phis.ndim == 2 and self.free_energy.num_comps == self.phis.shape[1]

    def set_random_composition(
        self,
        dist: str = "simplex_uniform",
        *,
        num_phases: int = None,
        rng=None,
        **kwargs,
    ):
        """choose random compositions for the phases

        Args:
            dist: Determines the distribution determining the weights of the components
            num_phases: The number of phases. If omitted, the number is not changed.
            rng: The random number generator
        """
        super().set_random_composition(dist, num_phases=num_phases, rng=rng, **kwargs)
        assert self.is_consistent

    @property
    def free_energy_densities(self) -> np.ndarray:
        """return the free energy densities of all phases"""
        return self.free_energy.free_energy_density(self.phis)

    @property
    def chemical_potentials(self) -> np.ndarray:
        """return the chemical potentials of all phases and components"""
        return self.free_energy.chemical_potentials(self.phis)

    @property
    def pressures(self) -> np.ndarray:
        """return the pressures of all phases"""
        return self.free_energy.pressure(self.phis)

    @property
    def entropy_production(self) -> float:
        """float: the estimated entropy production in the system"""
        mus = self.chemical_potentials
        ps = self.pressures
        dmu = (mus[:, np.newaxis, :] - mus[np.newaxis, :, :]) ** 2
        dp = (ps[:, np.newaxis] - ps[np.newaxis, :]) ** 2
        return float(np.einsum("ni,mi,nmi->", self.phis, self.phis, dmu) + dp.sum())

    def at_equilibrium(self, tol: float = 1e-5):
        """checks whether the multiphase system is at equilibrium"""
        return self.entropy_production < tol


class MultiphaseVolumeSystem(MultiphaseSystem):
    """represents a multicomponent system of multiple phases"""

    def __init__(
        self, free_energy: FloryHuggins, phis: np.ndarray, volumes: np.ndarray
    ):
        """
        Args:
            free_energy: The multicomponent free energy
            phis: The composition vectors for each phase (a 2d array)
            volumes: The volumes of all compartments
        """
        self.volumes = np.asanyarray(volumes)
        super().__init__(free_energy, phis)

    @classmethod
    def from_amounts(
        cls, free_energy: FloryHuggins, phis: np.ndarray, amounts: np.ndarray
    ) -> MultiphaseVolumeSystem:
        """construct system with volumes from composition and total amounts of material

        Args:
            free_energy: The multicomponent free energy
            phis: The composition vectors for each phase (a 2d array)
            amounts: The total amounts of all components
        """
        phis = np.asanyarray(phis)
        assert len(amounts) == phis.shape[1]
        volumes = np.linalg.lstsq(phis.T, amounts, rcond=None)[0]
        return cls(free_energy, phis.copy(), volumes)

    @classmethod
    def from_random_composition(  # type: ignore
        cls,
        free_energy: FloryHuggins,
        volumes: np.ndarray,
        dist: str = "simplex_uniform",
        *,
        rng=None,
        **kwargs,
    ) -> MultiphaseSystem:
        """create multiphase system with random compositions

        Args:
            free_energy: The multicomponent free energy
            volumes: The volumes of all compartments
            dist: Determines the distribution determining the weights of the components
            rng: The random number generator
        """
        num_phases = len(volumes)
        obj = cls(
            free_energy=free_energy,
            phis=np.zeros((num_phases, free_energy.num_comps)),
            volumes=volumes,
        )
        obj.set_random_composition(dist=dist, rng=rng, **kwargs)
        return obj

    def copy(
        self, phis: np.ndarray = None, volumes: np.ndarray = None
    ) -> MultiphaseVolumeSystem:
        """copy the system (with potentially a new composition)"""
        if phis is None:
            phis = self.phis
        if volumes is None:
            volumes = self.volumes
        return self.__class__(self.free_energy, phis.copy(), volumes.copy())

    @property
    def is_consistent(self) -> bool:
        """does some self-consistency checks"""
        return super().is_consistent and len(self.volumes) == self.phis.shape[0]

    @property
    def total_volume(self) -> float:
        """float: total volume of the system"""
        return self.volumes.sum()  # type: ignore

    @property
    def phi_bar(self) -> np.ndarray:
        """:class:`numpy.ndarray`: Mean fraction averaged over phases"""
        return (self.volumes @ self.phis) / self.total_volume  # type: ignore

    @property
    def amounts(self) -> np.ndarray:
        """return the amounts of all components (including the solvent) in all phases"""
        return self.volumes[:, np.newaxis] * self.phi_all  # type: ignore

    @property
    def total_amounts(self) -> np.ndarray:
        """return the total amounts of all components (including the solvent)"""
        return self.amounts.sum(axis=0)  # type: ignore

    @property
    def free_energy_value(self) -> float:
        """total free energy"""
        return float(self.volumes @ self.free_energy_densities)
