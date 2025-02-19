"""
Module defining thermodynamic quantities of multicomponent phase separation.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from __future__ import annotations

import logging
from typing import Callable

import numba as nb
import numpy as np
import scipy.linalg
from scipy.cluster import hierarchy
from scipy.spatial import distance


class SolventFractionError(RuntimeError):
    """error indicating that the solvent fraction was not in [0, 1]"""

    pass


class FloryHuggins:
    r"""represents the free energy of a multicomponent mixture

    The particular implementation of the free energy density reads

    .. math::
        f(\{\phi_i\}) = \frac{k_\mathrm{B}T}{\nu}\biggl[
            \phi_0\ln(\phi_0)
            + \sum_{i=1}^N \phi_i \ln(\phi_i)
            + \!\sum_{i,j=1}^N \frac{\chi_{ij}}{2} \phi_i\phi_j
        \biggr]

    where :math:`\phi_i` is the fraction of component :math:`i` and
    :math:`\phi_0 = 1 - \sum_i \phi_i` is the fraction of the solvent. All components
    are assumed to have the same molecular volume :math:`\nu` and the interactions are
    quantified by the Flory matrix :math:`\chi_{ij}`. Note that components do not
    interact with the solvent, which is thus completely inert.
    """

    def __init__(self, chis: np.ndarray, *, inert_solvent: bool = True):
        """
        Args:
            chis (:class:`~numpy.ndarray`):
                The interaction matrix
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
        """
        if isinstance(chis, FloryHuggins):
            chis = chis.chis  # type: ignore
        self.inert_solvent = inert_solvent
        chis = np.atleast_1d(chis)
        num_comps = chis.shape[0]
        shape = (num_comps, num_comps)

        chis = np.array(np.broadcast_to(chis, shape))

        if self.inert_solvent:
            if not np.allclose(np.diag(chis), 0):
                logging.warning("Diagonal of chi matrix is not used for inert solvent")
            # ensure that the diagonal entries vanish
            np.fill_diagonal(chis, 0)

        # ensure that the chi matrix is symmetric
        if not np.allclose(chis, chis.T):
            logging.warning("Using symmetrized χ interaction-matrix")
        self.chis = 0.5 * (chis + chis.T)

    @property
    def num_comps(self) -> int:
        """int: the number of components (without the solvent) in the mixture"""
        return int(self.chis.shape[0])

    @property
    def independent_entries(self) -> np.ndarray:
        """:class:`~numpy.ndarray` entries of the upper triangle only"""
        if self.inert_solvent:
            return self.chis[np.triu_indices_from(self.chis, k=1)]  # type: ignore
        else:
            return self.chis[np.triu_indices_from(self.chis, k=0)]  # type: ignore

    def copy(self) -> FloryHuggins:
        return self.__class__(self.chis, inert_solvent=self.inert_solvent)

    @classmethod
    def from_uniform(
        cls, num_comp: int, chi: float, *, inert_solvent: bool = True
    ) -> FloryHuggins:
        """create Flory-Huggins free energy with uniform chi matrix

        Args:
            num_comp (int):
                The number of components
            chi (float):
                The value of all non-zero values in the interaction matrix
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
        """
        chis = np.full((num_comp, num_comp), chi)
        if inert_solvent:
            chis[np.diag_indices_from(chis)] = 0
        return cls(chis, inert_solvent=inert_solvent)

    @classmethod
    def from_random_normal(
        cls,
        num_comp: int,
        chi_mean: float = 0,
        chi_std: float = 1,
        *,
        inert_solvent: bool = True,
        rng=None,
    ) -> FloryHuggins:
        """create random Flory-Huggins free energy density

        Args:
            num_comp (int):
                Number of components (excluding the solvent)
            chi_mean (float):
                Mean interaction
            chi_std (float):
                Standard deviation of the interactions
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
            rng:
                the random number generator
        """
        obj = cls(np.zeros((num_comp, num_comp)), inert_solvent=inert_solvent)
        obj.set_random_chis(chi_mean, chi_std, rng=rng)
        return obj

    @classmethod
    def from_block_structure(
        cls,
        num_comp: int,
        num_blocks: int,
        chi_attract: float,
        chi_repel: float = 0,
        noise: float = 0,
        *,
        inert_solvent: bool = True,
        rng=None,
    ) -> FloryHuggins:
        """create random Flory-Huggins free energy density with block structure

        Args:
            num_comp (int):
                Number of components (excluding the solvent)
            chi_attract (float):
                Mean value of the attractive interaction
            chi_repel (float):
                Mean value of the repulsive interaction
            noise (float):
                Normalized standard deviation of the perturbations of the entries
            inert_solvent (bool):
                Flag determining whether the solvent (species 0) is assumed inert or not.
                For an inert solvent, the diagonal of the `chi` matrix must vanish.
            rng:
                the random number generator
        """
        obj = cls(np.zeros((num_comp, num_comp)), inert_solvent=inert_solvent)
        obj.set_block_structure(num_blocks, chi_attract, chi_repel, noise, rng=rng)
        return obj

    def remove_component(self, i: int, *, inplace: bool = False) -> FloryHuggins:
        """create free energy with one component removed

        Args:
            i (int): Index of the component that is removed
            inplace (bool): Change this free energy instead of returning a new one
        """
        assert self.num_comps > 1
        chis = np.delete(np.delete(self.chis, i, 0), i, 1)

        if inplace:
            self.chis = chis
            return self
        else:
            return self.__class__(chis)

    def duplicate_component(self, i: int, *, inplace: bool = False) -> FloryHuggins:
        """create free energy with one component duplicated

        Args:
            i (int): Index of the component that is duplicated
            inplace (bool): Change this free energy instead of returning a new one
        """
        chis = np.zeros((self.num_comps + 1, self.num_comps + 1))
        chis[: self.num_comps, : self.num_comps] = self.chis
        chis[-1, : self.num_comps] = self.chis[i, :]
        chis[: self.num_comps, -1] = self.chis[:, i]

        if inplace:
            self.chis = chis
            return self
        else:
            return self.__class__(chis)

    def reorder_components(self) -> None:
        """reorder the components by pairing similar components"""
        dists = distance.pdist(self.chis)
        link = hierarchy.linkage(dists, optimal_ordering=True)[:, :2].astype(int)
        order = []

        def append_link(a):
            """helper function for determining the new order"""
            if a < self.num_comps:
                order.append(a)
            else:
                append_link(link[a - self.num_comps, 0])
                append_link(link[a - self.num_comps, 1])

        append_link(link[-1, 0])
        append_link(link[-1, 1])

        assert list(range(self.num_comps)) == list(sorted(order))
        self.chis = self.chis[order, :][:, order]

    def set_random_chis(self, chi_mean: float = 0, chi_std: float = 1, *, rng=None):
        """choose random interaction parameters

        Args:
            chi_mean: Mean interaction
            chi_std: Standard deviation of the interactions
            rng: the random number generator
        """
        if rng is None:
            rng = np.random.default_rng()

        self.chis[:] = 0  # reset old values

        # determine random entries
        if self.inert_solvent:
            num_entries = self.num_comps * (self.num_comps - 1) // 2
        else:
            num_entries = self.num_comps * (self.num_comps + 1) // 2
        chi_vals = rng.normal(chi_mean, chi_std, num_entries)

        # build symmetric  matrix from this
        i, j = np.triu_indices(self.num_comps, 1 if self.inert_solvent else 0)
        self.chis[i, j] = chi_vals
        self.chis[j, i] = chi_vals

    def set_block_structure(
        self,
        num_blocks: int,
        chi_attract: float,
        chi_repel: float = 0,
        noise: float = 0,
        *,
        rng=None,
    ) -> None:
        """choose random interaction parameters with block structure

        Note:
            The solvent is always assumed to be inert in this case. Only the noise is
            applied to all components equally.

        Args:
            num_comp: Number of components (excluding the solvent)
            chi_attract: Mean value of the attractive interaction
            chi_repel : Mean value of the repulsive interaction
            noise: Normalized standard deviation of the perturbations of the entries
            rng: the random number generator
        """
        num_comps = self.num_comps
        block_size = int(np.ceil(num_comps / num_blocks))
        # get dimensions of the individual blocks
        dims = np.diff(np.r_[np.arange(0, num_comps, block_size), num_comps])

        # determine the interaction matrix
        self.chis[:] = (1 - np.eye(num_comps)) * chi_repel
        i = scipy.linalg.block_diag(*[(1 - np.eye(d)) for d in dims])
        self.chis[i == 1] = chi_attract

        # add noise
        if noise != 0:
            if rng is None:
                rng = np.random.default_rng()
            if self.inert_solvent:
                factors = rng.normal(1, noise, size=num_comps * (num_comps - 1) // 2)
                i, j = np.triu_indices(num_comps, 1)
            else:
                factors = rng.normal(1, noise, size=num_comps * (num_comps + 1) // 2)
                i, j = np.triu_indices(num_comps, 0)
            self.chis[i, j] *= factors
            self.chis[j, i] *= factors

    @property
    def num_entries(self) -> int:
        """int: the number of independent entries in the interaction matrix"""
        if self.inert_solvent:
            return self.num_comps * (self.num_comps - 1) // 2
        else:
            return self.num_comps * (self.num_comps + 1) // 2

    @property
    def chis_flat(self) -> np.ndarray:
        """return (a copy of) all independent entries of the matrix"""
        if self.inert_solvent:
            i, j = np.triu_indices(self.num_comps, 1)
        else:
            i, j = np.triu_indices(self.num_comps, 0)
        return self.chis[i, j]  # type: ignore

    @chis_flat.setter
    def chis_flat(self, values: np.ndarray) -> None:
        """sets all independent entries of the matrix"""
        if self.inert_solvent:
            i, j = np.triu_indices(self.num_comps, 1)
        else:
            i, j = np.triu_indices(self.num_comps, 0)
        self.chis[i, j] = values
        self.chis[j, i] = values

    @property
    def chis_full(self) -> np.ndarray:
        """return the full chi matrix including the solvent"""
        n = self.num_comps
        result = np.zeros((n + 1, n + 1))
        result[:n, :n] = self.chis
        return result

    @property
    def is_consistent(self) -> bool:
        """bool: return whether the chi matrix is symmetric and trace-free"""
        diag = np.diag(self.chis)
        if self.inert_solvent:
            return np.allclose(self.chis, self.chis.T) and np.allclose(diag, 0)
        else:
            return np.allclose(self.chis, self.chis.T)

    def add(self, i: int, j: int, value: float):
        """adds a value to the specified position"""
        if self.inert_solvent and i == j and value != 0:
            raise ValueError("Cannot alter the diagonal")
        self.chis[i, j] += value
        if i != j:
            self.chis[j, i] += value

    def add_flat(self, flat_idx: int, value: float):
        """adds a value to the position specified by a flat index"""
        # TODO: this could be sped up by calculating the indices directly
        if self.inert_solvent:
            i, j = np.triu_indices(self.num_comps, 1)
        else:
            i, j = np.triu_indices(self.num_comps, 0)
        i, j = i[flat_idx], j[flat_idx]
        self.chis[i, j] += value
        if i != j:
            self.chis[j, i] += value

    def free_energy_density(
        self, phis: np.ndarray, *, check: bool = True
    ) -> np.ndarray:
        """returns free energy for a given composition

        Args:
            phis (:class:`numpy.ndarray`): The composition of the phase(s)
            check (bool): Whether the solvent fraction is checked to be positive
        """
        phis = np.asanyarray(phis)
        assert phis.shape[-1] == self.num_comps, "Wrong component count"

        phi_sol = 1 - phis.sum(axis=-1)
        if check and np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")

        entropy_comp = np.einsum("...i,...i->...", phis, np.log(phis))
        entropy_sol = phi_sol * np.log(phi_sol)
        enthalpy = 0.5 * np.einsum("...i,...j,ij->...", phis, phis, self.chis)
        return entropy_comp + entropy_sol + enthalpy  # type: ignore

    def chemical_potentials(self, phis: np.ndarray) -> np.ndarray:
        """returns chemical potentials for a given composition"""
        phis = np.asanyarray(phis)
        phi_sol = 1 - phis.sum(axis=-1, keepdims=True)
        if np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")
        return (  # type: ignore
            np.log(phis) - np.log(phi_sol) + np.einsum("...i,ij->...j", phis, self.chis)
        )

    def pressure(self, phis: np.ndarray) -> np.ndarray:
        """returns pressure for a given composition"""
        f = self.free_energy_density(phis)
        mus = self.chemical_potentials(phis)
        return np.einsum("...i,...i->...", phis, mus) - f  # type: ignore

    def hessian(self, phis: np.ndarray) -> np.ndarray:
        """returns Hessian for the given composition"""
        phis = np.asanyarray(phis)
        assert phis.shape == (self.num_comps,)
        phi_sol = 1 - phis.sum(axis=-1, keepdims=True)
        if np.any(phi_sol < 0):
            raise SolventFractionError("Solvent has negative concentration")
        return np.eye(self.num_comps) / phis + 1 / phi_sol + self.chis  # type: ignore

    def unstable_modes(self, phis: np.ndarray) -> int:
        """returns the number of unstable modes"""
        eigenvalues = np.linalg.eigvalsh(self.hessian(phis))
        return int(np.sum(eigenvalues < 0))

    def is_stable(self, phis: np.ndarray) -> bool:
        """checks whether a given composition is (linearly) stable"""
        return self.unstable_modes(phis) == 0

    def make_calc_vars(self) -> Callable:
        """returns a compiled function calculating variables for a given composition

        Calling the function with a composition vector will result in a tuple consisting
        of the associated free energy, chemical potentials, and pressure.
        """
        num_comps = self.num_comps
        chis_default = self.chis
        include_diagonal = not self.inert_solvent

        @nb.njit
        def calc_vars(phis: np.ndarray, chis: np.ndarray = None):
            """calculates free energy, chemical potential, and pressure"""
            if chis is None:
                chis = chis_default

            phi_sol = 1 - phis.sum()
            if phi_sol < 0:
                raise SolventFractionError("Solvent has negative concentration")

            log_phi_sol = np.log(phi_sol)
            f = phi_sol * log_phi_sol  # entropy of solvent
            mu = np.empty(num_comps)
            p = 0
            for i in range(num_comps):  # iterate components
                f += phis[i] * np.log(phis[i])  # entropy of component i
                for j in range(i):
                    f += chis[i, j] * phis[i] * phis[j]
                if include_diagonal:
                    f += 0.5 * chis[i, i] * phis[i] ** 2
                mu[i] = np.log(phis[i]) - log_phi_sol + chis[i] @ phis
                p += phis[i] * mu[i]
            p -= f

            return f, mu, p

        return calc_vars  # type: ignore

    def make_calc_diffs(self) -> Callable:
        """returns a compiled function calculating variables without constant terms

        This is useful for calculating differences.
        """
        chis_default = self.chis

        @nb.njit
        def calc_diffs(phis: np.ndarray, chis: np.ndarray = None):
            """calculates chemical potential and pressure"""
            if chis is None:
                chis = chis_default

            phi_sol = 1 - phis.sum()
            if phi_sol < 0:
                raise SolventFractionError("Solvent has negative concentration")

            log_phi_sol = np.log(phi_sol)
            mu = np.log(phis)
            p = -log_phi_sol
            for i in range(len(phis)):  # iterate over components
                val = chis[i] @ phis
                mu[i] += val - log_phi_sol
                p += 0.5 * val * phis[i]

            return mu, p

        return calc_diffs  # type: ignore
