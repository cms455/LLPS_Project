"""
Module to utilize the flory library to perform physical computing using LLPS
'Flory' performs an evolutionary algorithm to simulate N-component systems
"""

import sys
import itertools
import numpy as np
from tqdm.notebook import tqdm
from importlib import reload
import matplotlib as mpl
import matplotlib.pyplot as plt
import mpltern
from concurrent.futures import ProcessPoolExecutor

# CUSTOM LIBRARY
#import flory.flory as flory # Ref: https://github.com/qiangyicheng/flory/tree/main
import flory
import alps.graph as graph
import alps.utilities as ut


def get_random_phi(ncomp):
    """
    Return a random composition
    """
    phi = np.random.random(ncomp)
    phi /= np.sum(phi)
    return phi

def get_phis(nComp=2, n=11, prec=1e-6):
    """
    Returns an array of volume fractions of n species.

    Parameters
    ----------
    n : int, optional
        Number of grid points along one ternary axis, by default 11
    prec : float, optional
        Tolerance for triangular points, by default 1e-6

    Returns
    -------
    (t, l, r) : tuple[np.ndarray]
        Ternary coordinates.
    """
    # top axis in descending order to start from the top point
    t = np.linspace(1, 0, n)
    points = []
    for tmp in itertools.product(t, repeat=nComp):
        if abs(sum(tmp) - 1.0) > prec:
            continue
        points.append(tmp)
    points = np.array(points)
    return points

# Run a simulation
def process_phi(phi, chi, options={'progress': False}, return_finder=False):
    """
    Returns a sim result  for a given phi.

    Parameters
    ----------
    phi: np.ndarray, shape (num_comp,), volume fractions
    num_comp: int, number of components
    chi: np.ndarray, shape (num_comp, num_comp), chi matrix
    options: dict, options for flory.CoexistingPhasesFinder
        ... Example:
            options = {
                "num_part": 8, # number of partitions
                "progress": False, # no progress bar
                "max_steps": 10000000, # maximum number of steps
            }
    return_finder: bool, return CoexistingPhasesFinder object, by default False
    Returns
    -------
    phases: flory.flory.common.phases.Phases
    """
    num_comp = len(phi)
    free_energy = flory.FloryHuggins(num_comp, chi)
    interaction = free_energy.interaction
    entropy = free_energy.entropy
    ensemble = flory.CanonicalEnsemble(num_comp, phi)

    sim = flory.CoexistingPhasesFinder(
        interaction,
        entropy,
        ensemble,
        **options,
    )
    phases = sim.run().get_clusters()
    if return_finder:
        return phases, sim
    else:
        return phases

def phase2phis(phase, n=100, sort_by='volume'):
    """
    Convert a phase object to a phi array.
    Caution: This is an approximation. Raw info is in CoexistingPhasesFinder object.

    Parameters
    ----------
    phase: flory.flory.common.phases.Phases
        ... 'phases' is a class object that contains the results of the simulations
        ... phases.fractions: np.ndarray, shape (n, num_comp), volume fractions (phis)
        ... phases.volumes: np.ndarray, shape (n, num_comp), volumes
        ... phases.num_phases: int, number of phases
    n: int, number of compartments of output, by default 100
    sort_by: str, sorting option, by default 'volume'
        ... 'volume': sort by volume
        ... 'phi': sort by phi - the most concentrated component
        ... otherwise, do not sort
    """
    num_part = n # number of partitions
    num_comp = phase.num_components # number of components

    phis = phase.fractions
    volumes = phase.volumes/(np.sum(phase.volumes))

    if sort_by == 'volume':
        [volumes, phis] = ut.sortNArrays([volumes, phis], element_dtype=np.ndarray)
    elif sort_by == 'phi':
        phase = phase.sort() # sort by the most concentrated component
    else:
        pass
    phis = np.empty((num_comp, num_part))
    nPartitions = [round(volume * num_part) for volume in volumes]
    nPartitions[-1] = num_part - sum(nPartitions[:-1]) # adjust the last partition to add up to num_part
    starts, ends = np.cumsum([0] + nPartitions[:-1]), np.cumsum(nPartitions)
    for i, (start, end) in enumerate(zip(starts, ends)):
        phis[:, start:end] = np.repeat(phase.fractions[i][:, np.newaxis], end-start, axis=1)

    return phis





# Scan phase space (Serial version)
def scan_phase_space(num_comp, chi, n=11, verbose=True,
                     options={'progress': False}):
    """
    Scan a phase space with a chi matrix.

    Parameters
    ----------
    num_comp: int, number of components
    chi: np.ndarray, chi matrix, shape (num_comp, num_comp)

    Returns
    -------
    phis: np.ndarray, shape (n, num_comp), volume fractions
    res: list of 'phases' (flory.flory.common.phases.Phases object)
        ... 'phases' is a class object that contains the results of the simulations
        ... phases.fractions: np.ndarray, shape (n, num_comp), volume fractions (phis)
        ... phases.volumes: np.ndarray, shape (n, num_comp), volumes
        ... phases.num_phases: int, number of phases
    """

    # Pick an initial starting point
    phis = get_phis(nComp=num_comp, n=n)

    res = []  # results: list of phases (flory.flory.common.phases.Phases object)

    for i, phi in enumerate(tqdm(phis)):
        if verbose and i % 50 == 0:
            print(f'Step: {i} / {len(phis)}...')

        # Initialize
        free_energy = flory.FloryHuggins(num_comp, chi)
        interaction = free_energy.interaction
        entropy = free_energy.entropy
        ensemble = flory.CanonicalEnsemble(num_comp, phi)

        sim = flory.CoexistingPhasesFinder(  # flory.flory.mcmp.finder.CoexistingPhasesFinder
            interaction,
            entropy,
            ensemble,
            **options,
        )
        phases = sim.run().get_clusters()
        res.append(phases)
    return phis, res

# Scan phase space (Parallel version)
def scan_phase_space_pl(num_comp, chi, n=11, options={'progress': False}):
    """
    Scan a phase space with a chi matrix (parallel version).

    Parameters
    ----------
    num_comp: int, number of components
    chi: np.ndarray, shape (num_comp, num_comp)
    n: int, number of grid points along one ternary axis, by default 11
    options: dict, options for flory.CoexistingPhasesFinder
        ... Example:
            options = {
                "num_part": 8, # number of partitions
                "progress": False, # no progress bar
                "max_steps": 1,000,000, # maximum number of steps of the simulation per point
            }
    Returns
    -------
    phis: np.ndarray, shape (n, num_comp), volume fractions
    res: list of 'phases' (flory.flory.common.phases.Phases object)
        ... 'phases' is a class object that contains the results of the simulations
        ... phases.fractions: np.ndarray, shape (n, num_comp), volume fractions (phis)
        ... phases.volumes: np.ndarray, shape (n, num_comp), volumes
        ... phases.num_phases: int, number of phases
    """
    phis = get_phis(nComp=num_comp, n=n)
    res = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_phi, phi, chi, options) for phi in phis]
        for i, future in enumerate(tqdm(futures)):
            res.append(future.result())

    return phis, res




def plot_ternary_phase_diagram(phis, slns, fignum=1, figsize=(6, 6), subplot=111,
                               cb_kwargs={'fraction': 0.025,'pad': 0.1, 'fontsize': 10},):
    """
    Plot a ternary phase diagram.

    Parameters
    ----------
    phis
    slns
    fignum
    figsize
    subplot

    Returns
    -------

    """
    nComp = len(phis[0])
    nPhases = [s.num_phases for s in slns]

    fig, ax = graph.set_fig(fignum, figsize=figsize, subplot=subplot, projection='ternary')
    if max(nPhases) < 10:
        cmap, norm = graph.get_discrete_cmap_norm([f'C{i - 1:02d}' for i in range(1, max(nPhases) + 1)], vmin=1, vmax= max(nPhases))
        colors = [plt.get_cmap('tab10')(i) for i in range(max(nPhases))]
    else:
        colors = graph.get_color_from_cmap(cmap='turbo', n=10)
        cmap, norm = graph.get_discrete_cmap_norm(colors, vmin=1, vmax=max(nPhases))

    t, l, r = phis[:, 0], phis[:, 1], phis[:, 2]
    ax.tripcolor(t, l, r, nPhases, cmap=cmap, norm=norm)

    ax.set_aspect(1)
    graph.add_discrete_colorbar(ax, colors, vmin=1, vmax=nComp + 1,
                                label='No. of Phases', **cb_kwargs )

    ax.set_tlabel('$\phi_1$')
    ax.set_llabel('$\phi_2$')
    ax.set_rlabel('$\phi_3$')

    return fig, ax



