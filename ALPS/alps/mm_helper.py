#!/usr/bin/env python3
"""
Helper for multicomponent_evolution.py
"""

from typing import List, Tuple

import numpy as np
from numba import njit
from scipy import cluster, spatial

import graph as graph
import matplotlib.pyplot as plt
import matplotlib as mpl
from multicomponent_evolution_takumi import *



DT_INITIAL: float = 1.0  # initial time step for the relaxation dynamics
TRACKER_INTERVAL: float = 10.0  # interval for convergence check
TOLERANCE: float = 1e-4  # tolerance used to decide when stationary state is reached

CLUSTER_DISTANCE: float = 1e-2  # cutoff value for determining composition clusters

PERFORMANCE_TOLERANCE: float = 0.5  # tolerance used when calculating performance
KILL_FRACTION: float = 0.3  # fraction of population that is replaced each generation

REPETITIONS: int = 64  # number of samples used to estimate the performance


def isPhiPhysical(phis):
    """
    Checks if the volume fraction in each layer is physical

    Parameters
    ----------
    phis: np.ndarray or 1D array, volume fraction of all phases

    Returns
    -------
    bool: True if the volume fraction is physical, False otherwise
    """
    if np.sum(phis) > 1:
        return False
    else:
        for phi in phis:
            if phi < 0 or phi > 1:
                return False
        return True

def fix_phis(phis):
    """
    Normalize the volume fractions to sum to 1 and clip to [0, 1]

    Parameters
    ----------
    phis: np.ndarray or 1D array, volume fraction of all phases
        ... Shape (num_phases, num_comps) or (num_comps,)

    Returns
    -------
    phis: np.ndarray, Normalized volume fraction of all phases
        ... Shape (num_phases, num_comps) or (num_comps,)
    """
    original_shape = phis.shape
    if len(phis.shape) == 1:
        phis = phis.reshape(1, len(phis)) # num_phases x num_comps
    nphase, ncomp = phis.shape
    for n in range(nphase):
        # Clip the volume fractions to [0, 1]
        for d in range(ncomp):
            if phis[n, d] < 0:
                phis[n, d] = 0
            if phis[n, d] > 1:
                phis[n, d] = 1
        # If phi does not add up to 1, that is okay as long as it is less than 1. Solvent makes up the rest in this case.
        # However, if phi is greater than 1, normalize the volume fractions, assuming that there is no solvent
        if np.sum(phis[n, :]) > 1:
            # Normalize the volume fractions to sum to 1 for each layer
            phis[n, :] /= np.sum(phis[n, :])
    return phis.reshape(original_shape)


def get_composition_gaussian(num_phases, avg_phis, phi_std=0.01, phi_std_min=1e-6, verbose=True):
    """
    Generate volume fractions of all phases with Gaussian noise around the average composition

    Parameters
    ----------
    num_phases: int, number of phases
    avg_phis: np.ndarray or 1D array, average composition of all components
    phi_std: float, standard deviation of the Gaussian noise
    phi_std_min: float, minimum standard deviation of the Gaussian noise
    verbose: bool, print the average composition of the generated volume fractions

    Returns
    -------
    phis: np.ndarray, volume fractions of all phases
        ... Shape (num_phases, num_comps)
    """
    if not isPhiPhysical(avg_phis):
        raise ValueError("The average composition is not physical. "
                         "Volume fractions should be between 0 and 1 and sum to 1.")
    # If phi_std is too small, we will get stuck in a local minimum
    # Return the average composition
    if phi_std < phi_std_min:
        phis = np.repeat(avg_phis, num_phases).reshape(len(avg_phis), num_phases).T
        return phis

    avg_phis = np.asarray(avg_phis)
    num_comps = len(avg_phis)

    phis = np.zeros((num_phases, num_comps))
    for n in range(num_phases-1):
        for d in range(num_comps):
            # phi_d: the composition of the d-th component in the n-th phase
            phi_d = np.random.normal(avg_phis[d], scale=phi_std)
            phi_max = 1 - np.sum(phis[n, :d]) # the maximum value of phi_d
            phi_d = max(0, min(phi_d, phi_max)) # clip to [0, Maximal Allowed Value]
            phis[n, d] = phi_d
        # # The last component is determined by the sum of the other components
        # phis[n, -1] = 1 - np.sum(phis[n, :-1])
    phis[-1, :] = num_phases * avg_phis - np.sum(phis[:-1, :], axis=0) # the last phase is made to achive the average composition
    phis = fix_phis(phis)
    if verbose:
        print(f'Composition (phi vector): {np.mean(phis, axis=0)}')
    return phis


# VISUALIZING FUNCTIONS
def visualize_color(color, strip_height=1, strip_width=10):
    """
    Visualizes a given color as a horizontal strip.

    Parameters:
    - color: An array-like object representing the RGB values of the color (each value between 0 and 1).
    - strip_height: The height of the strip in inches (default is 1 inch).
    - strip_width: The width of the strip in inches (default is 10 inches).
    """
    color = np.asarray(color)
    if len(color.shape) == 1:
        color = color.reshape(1, len(color))

    nphase, ncomp = color.shape

    fig, ax = plt.subplots(figsize=(strip_width, strip_height * nphase))
    for i in range(nphase):
        ax.fill_between([0, 1], 0+i, 1+i, color=color[i])
    ax.axis('off')  # Turn off the axes
    return fig, ax


def visualize_mixture(phis, height=1, width=1, fignum=1, subplot=111, ax=None, cmap='turbo',
                      solvent_color='powderblue', gamma=1.0):
    """
    Visualizes the volume fractions of all components in a single layer.

    Parameters
    ----------
    phis
    height
    width
    fignum
    subplot
    ax
    cmap
    solvent_color

    Returns
    -------

    """

    def create_n_cmap_bases(n, cmap='turbo', vmin=0., vmax=1, solvent_color='powderblue'):
        """
        Returns a list of n colormaps, each of which scans from solvent_color to a different color in the colormap.
        Parameters
        ----------
        n
        cmap
        vmin: int, minimum value of the colormap, choose from 0 to 1
        vmax: int, maximum value of the colormap, choose from 0 to 1

        Returns
        -------

        """
        if isinstance(cmap, str):
            cmap = plt.get_cmap(cmap)
        # Grub n colors from a given colormap between vmin and vmax, call them basis_colors
        basis_colors = [cmap(val) for val in np.linspace(vmin, vmax, n)]  # rgb values

        # create a linear color map
        colors_list = [graph.get_color_list_gradient(color1=solvent_color, color2=graph.rgb2hex(basis_color), ) for basis_color
                       in basis_colors]
        cmaps = [graph.create_cmap_from_colors(colors) for colors in colors_list]
        return cmaps

    def assign_color(phi, cmap='turbo', solvent_color='powderblue', gamma=1):
        """
        Assign a color to a given volume fraction using a colormap.

        Parameters
        ----------
        phi: 1d array, volume fraction of each component in a single layer
        cmap: str or colormap object, colormap to use

        Returns
        -------
        color: 1d array, RGBA values of the color
        """
        phi = np.power(phi, gamma) # Exggerate the color
        n = len(phi)  # number of components
        # cmaps = [truncate_colormap(cmap, minval=i/n, maxval=(i+1)/n) for i in range(n)]
        cmaps = create_n_cmap_bases(n, cmap=cmap, vmin=0., vmax=1, solvent_color=solvent_color)
        colors = np.empty((4, len(phi)))
        for i in range(n):
            colors[:, i] = cmaps[i](phi[i])
        # sum colors
        color = np.mean(colors, axis=1) # average over all components

        # # Optional: Set `1 - solvent concentration` as the alpha- bad idea since it makes the color too transparent
        # color[-1] = 1 - np.sum(phi)
        return color

    nphases, ncomps = phis.shape
    colors = []
    for i in range(nphases):
        phi = phis[i, :] # volume fraction of each phase
        color = assign_color(phi, cmap=cmap, solvent_color=solvent_color, gamma=gamma)
        colors.append(color)

    # Visualize:
    ## Set up Figure and Axes objects
    if ax is None:
        fig = plt.figure(num=fignum, figsize=(width, height))
        ax = fig.add_subplot(subplot)
    else:
        fig = ax.get_figure()

    for i in range(nphases):
        ax.fill_between([0, width], (height / nphases) * i, (height / nphases) * (i + 1),
                        color=colors[i], linewidth=0)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    # ADD EXTRA STRIP FOR SOLVENT
    # color_bkg = np.concatenate((graph.cname2rgba(solvent_color, normalize=False), [0.0]))  # solvent color
    # ax.fill_between([0, width], 0, height, color=color_bkg, linewidth=0)

    # ax.axis('off')  # Turn off the axes
    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    fig.tight_layout()

    # Return color keys for each component
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    basis_colors = [cmap(val) for val in np.linspace(0, 1, ncomps)]  # Color keys for each component
    return fig, ax, basis_colors

def visualize_chi(chi, ax=None, cmap='coolwarm', vmin=None, vmax=None, fontsize=12):
    """
    Visualize the interaction parameter matrix chi.

    Parameters
    ----------
    chi: np.ndarray, interaction parameter matrix
    ax: Axes object, axes to plot the chi matrix
    cmap: str, colormap to use for visualizing the interaction parameter matrix
    vmin: float, minimum value of chi
    vmax: float, maximum value of chi
    fontsize: int, fontsize of the title and labels

    Returns
    -------
    fig: Figure object
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()
    v_mid = 2.
    vdiff = max(np.max(chi) - v_mid, v_mid - np.min(chi))
    if vmin is None:
        vmin = max(v_mid - vdiff, 0)
    if vmax is None:
        vmax = round(v_mid + vdiff + 0.5)


    im = ax.imshow(chi, cmap=cmap, vmin=vmin, vmax=vmax, extent=[0.5, len(chi)+0.5, 0.5, len(chi)+0.5])

    fig.colorbar(im, ax=ax)
    ax.set_title('$\chi_{ij}$', fontsize=fontsize)
    ax.set_xlabel('Component $j$', fontsize=fontsize*0.9)
    ax.set_ylabel('Component $i$', fontsize=fontsize*0.9)

    ax.set_xticks(np.arange(len(chi))+1)
    ax.set_yticks(np.arange(len(chi))+1)
    ax.set_yticklabels(np.arange(len(chi))[::-1]+1)
    # ax.invert_yaxis()
    return fig, ax


def visualize_mixtures(list_of_phis, height=4, width=5, fignum=1, gamma=1,
                       cmap='rainbow', solvent_color='azure', #'powderblue' is also a good color for solvent
                       t=None, chi=None, cmap_chi='coolwarm',
                       fontsize=12):
    """
    Visualize the volume fractions of all phases in a mixture.

    Parameters
    ----------
    list_of_phis: list of np.ndarray, volume fractions of all phases
    height: float, height of the figure
    width: float, width of the figure
    fignum: int, figure number
    cmap: str or colormap object, colormap to use for visualizing the volume fractions
    solvent_color: str, color of the solvent
    t: list of float, time points
    chi: np.ndarray, interaction parameter matrix
    cmap_chi: str or colormap object, colormap to use for visualizing the interaction parameter matrix
    fontsize: int, fontsize of the title and labels

    Returns
    -------
    fig: Figure object
    """
    nphis = len(list_of_phis)
    if chi is not None:
        nphis += 1
    fig, axes = plt.subplots(num=fignum, ncols=nphis, figsize=(width * nphis, height))
    if chi is not None:
        axes_phi = axes[1:]
    else:
        axes_phi = axes

    for i, (phis, ax) in enumerate(zip(list_of_phis, axes_phi)):
        fig, ax, colors = visualize_mixture(phis, ax=ax, cmap=cmap, solvent_color=solvent_color, gamma=gamma)
    if t is not None:
        dt = t[1] - t[0]
        for t_, ax in zip(t, axes_phi):
            ax.set_title('$t=$' + f'{t_ / dt:.1f}' + '$\Delta t_{\\rm step}$', fontsize=12)

    if chi is not None:
        visualize_chi(chi, ax=axes[0], cmap=cmap_chi, fontsize=fontsize)

    phi_avg_true = np.mean(list_of_phis[0], axis=0) # True average composition
    phi_sol = 1 - np.sum(phi_avg_true)
    phi_avg_true = np.concatenate(([phi_sol], phi_avg_true)) # Add solvent composition
    phi_text = [f'{value:.2f}' for value in phi_avg_true]
    title = '$\\vec{\phi}=\{ \\phi_{sol}, \phi_i\}=$' + f'{phi_text}'.replace('[', '(').replace(']', ')').replace("'", '')
    graph.suptitle(title, fontsize=12, y=0.95)

    labels = ['$\phi_{sol}$']
    labels += [f'$\phi_{i}$' for i in range(1, len(phi_avg_true)+1)]
    colors = [solvent_color] + colors
    graph.legend_custom(axes_phi[0], colors, labels, ncol=min(len(phi_avg_true)+1, 5), loc='upper center', fontsize=fontsize, columnspacing=0.7, handlelength=0.5)

    return fig, axes

def main():
    pass

if __name__ == "__main__":
    main()
