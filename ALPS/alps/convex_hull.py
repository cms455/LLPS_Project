import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, delaunay_plot_2d, tsearch, ConvexHull, distance
import sympy as smat
from tqdm.notebook import tqdm
import matplotlib as mpl
import itertools
from numba import njit, prange
import time
import alps.graph as graph

# Basics
def get_phis(nComp=2, n=11, tol=1e-6):
    """
    Generates volume fractions for a given number of components, ensuring
    that all points sum to 1 within a specified tolerance.

    Parameters
    ----------
    n_species : int, optional
        Number of components (species), by default 2
    n_points : int, optional
        Number of grid points along one axis of the ternary diagram, by default 11
    tolerance : float, optional
        Numerical tolerance to ensure the sum of components is close to 1, by default 1e-6

    Returns
    -------
    np.ndarray
        An array of valid volume fractions where each row sums to 1 within the tolerance.
    """
    # Generate linearly spaced values between 0 and 1
    x = np.linspace(0, 1, n)
    # Create all possible combinations of these points for the given number of species
    all_combinations = np.array(list(itertools.product(x, repeat=nComp)))
    # Filter combinations where the sum is close to 1 within the specified tolerance
    points = all_combinations[np.abs(all_combinations.sum(axis=1) - 1) <= tol]
    return points

# Ternary system
def get_triangular_grid(n=11, prec=1e-6):
    """
    Generate a triangular grid

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
    for tmp in itertools.product(t, repeat=3):
        if abs(sum(tmp) - 1.0) > prec:
            continue
        points.append(tmp)
    points = np.array(points)
    return points[:, 0], points[:, 1], points[:, 2]

# FLORY-HUGGINS FREE ENERGY
# def getF(phi, chi, phi_thd=1e-5):
#     """
#     Get the Flory-Huggins free energy
#
#     Parameters
#     ----------
#     phi: np.ndarray, ternary coordinates, shape (n, m)
#     chi: np.ndarray, interaction matrix, shape (m, m)
#     phi_thd: float, threshold value for the volume fraction, by default 1e-5
#
#     Returns
#     -------
#     fe: np.ndarray, free energy
#     """
#     # Ensure that no element in phi is below phi_thd
#     phi2 = np.maximum(phi, phi_thd * np.ones_like(phi))
#
#     # Compute the first term: sum(phi * log(phi)) for each row
#     entropic_term = np.sum(phi2 * np.log(phi2), axis=1)
#
#     # Compute the second term: 1/2 * (phi @ chi @ phi.T) for each row
#     enthalpic_term = 0.5 * np.einsum('ij,ik,jk->i', phi, phi, chi)
#
#     # The free energy is the sum of the two terms
#     fe = entropic_term + enthalpic_term
#     return fe


@njit(parallel=True)
def getF(phi, chi, phi_thd=1e-5):
    """
    Get the Flory-Huggins free energy with parallel computation using numba

    Parameters
    ----------
    phi: np.ndarray, ternary coordinates, shape (n, m)
    ... CANNOT BE A 1D ARRAY. Reshape it to (1, -1) if it is a 1D array before passing it to this function
    chi: np.ndarray, interaction matrix, shape (m, m)
    phi_thd: float, threshold value for the volume fraction, by default 1e-5

    Returns
    -------
    fe: np.ndarray, free energy
    """
    # Ensure that no element in phi is below phi_thd
    phi2 = np.maximum(phi, phi_thd * np.ones_like(phi))

    # Initialize the free energy array
    n = phi.shape[0]
    fe = np.zeros(n)

    # Parallel computation of the free energy
    for i in prange(n):
        entropic_term = np.sum(phi2[i] * np.log(phi2[i]))
        enthalpic_term = 0.5 * np.sum(phi[i] * np.dot(chi, phi[i]))
        fe[i] = entropic_term + enthalpic_term

    return fe


# CHI MATRIX (INTERACTION MATRIX)
def get_chi_matrix_uniform(dim, val):
    """
    Create a uniform interaction matrix

    Example: dim=3
            chi = [[0, val, val],
                   [val, 0, val],
                   [val, val, 0]]

    Parameters
    ----------
    dim: int, dimension of the matrix
    val: float, value of the interaction

    Returns
    -------
    np.ndarray, the full, symmetric interaction matrix
    """
    chi = np.ones((dim, dim)) * val
    for i in range(dim):
        chi[i, i] = 0
    return chi


def get_random_chi(nComp, chi_mean=2, chi_std=0):
    """
    Create a random interaction matrix
    ... The non-diagonal elements are drawn from a normal distribution with mean chi_mean and standard deviation chi_std

    Args:
        num_comp (int): The component count
        chi_mean (float): The mean interaction strength
        chi_std (float): The standard deviation of the interactions

    Returns:
    chis: The full, symmetric interaction matrix
    """

    # initialize interaction matrix
    chis = np.zeros((nComp, nComp))

    # determine random entries
    num_entries = nComp * (nComp - 1) // 2
    chi_vals = np.random.normal(chi_mean, chi_std, num_entries)

    # build symmetric  matrix from this
    i, j = np.triu_indices(nComp, 1)
    chis[i, j] = chi_vals
    chis[j, i] = chi_vals
    return chis

# MAIN FUNCTIONS
def construct_adjacency_matrix(pts_simplex, thd):
    """
    Construct an adjacency matrix from a simplex based on whether an edge is stretched or not

    Parameters
    ----------
    pts_simplex: np.ndarray, points of a simplex
    thd: float, threshold value for the adjacency matrix, used to determine if an edge is stretched, must be positive

    Returns
    -------
    a: np.ndarray, adjacency matrix
    """

    def compute_distance(v1, v2):
        return np.sqrt(np.nansum((v2 - v1) ** 2))

    a = np.ones_like(pts_simplex) # a: Adjacency matrix
    dim = pts_simplex.shape[-1]
    for i in range(dim):
        for j in range(i, dim):
            if i < j:
                distance = compute_distance(pts_simplex[i, :], pts_simplex[j, :])
                isEdgeStretched = distance > thd  # this depends on nComp
                a[i, j] = a[j, i] = int(not isEdgeStretched)
    return a

def convex_hull(phi, fe):
    """
    Perform a convex hull on the free energy data
    ... For N-component system, the coordinate is (phi0, phi1, ..., phi_{N-1}, free_energy)


    Parameters
    ----------
    phi
    fe

    Returns
    -------

    """
    print('... Running convex hull')
    # For N-component system, the coordinate is (phi1, phi2, ..., phi_N, free_energy)
    ## You do not want to pass phi_N because you cannot run convex-hull when all data points are on the same facet
    pts = np.concatenate((phi[:, :-1], fe[:, np.newaxis]), axis=1) # phi_N is not passed. (n-1 dof) + 1 free energy
    ch = ConvexHull(pts)
    print('... Done')
    return ch, pts

def count_phases(ch, pts, phi_min_spacing, adjacency_tol=None, useHashTable=True):
    """

    Parameters
    ----------
    ch: scipy.spatial.ConvexHull
    pts: np.ndarray, ternary coordinates, shape (n, 3)
    phi_min_spacing: float, minimum spacing between points
    adjacency_tol: float, threshold value for the adjacency matrix, used to determine if an edge is stretched, must be positive
        ... If not provided, it is calculated as phi_min_spacing * np.sqrt(dim - 1) * 3
    useHashTable: bool, whether to use a hash table to store the number of phases for each adjacency matrix
        ... This is useful for large systems where the adjacency matrix is reused

    Returns
    -------
    nPhases: np.ndarray, number of phases in each simplex
    smpl_indices: np.ndarray, simplices
    """

    def make_hashtable(nComp):
        """
        A hash table that maps an adjacent matrix to a number of phases
        """

        def get_nPhase(a):
            """
            Get the number of phases from an adjacency matrix

            Parameters
            ----------
            a: np.ndarray, adjacency matrix, N x N

            Returns
            -------
            nPhase: int, number of phases, N x N
            """
            d = np.diag(np.sum(a, axis=1))
            eigDict = smat.Matrix(d - a).eigenvals()
            eigDict = {int(key): value for key, value in eigDict.items() if key == 0}
            nPhase = eigDict[0]
            return nPhase

        ht = {}
        a = np.ones((nComp, nComp))  # adjacent matrix

        combinations = itertools.product([0, 1], repeat=3)

        # Number of off-diagonal elements in the upper triangle
        num_off_diagonal = nComp * (nComp - 1) // 2

        # All combinations of 0s and 1s for the off-diagonal elements in the upper triangle
        combinations = itertools.product([0, 1], repeat=num_off_diagonal)

        matrices = []
        for combo in combinations:
            # Create an n x n matrix with 1s on the diagonal
            a = np.eye(nComp, dtype=int)

            # Fill the upper triangle off-diagonal elements
            combo_index = 0
            for i in range(nComp):
                for j in range(i + 1, nComp):
                    a[i, j] = combo[combo_index]
                    a[j, i] = combo[combo_index]  # Mirror to ensure symmetry
                    combo_index += 1
            nPhase = get_nPhase(a) # Get the number of phases
            ht[a.tobytes()] = int(nPhase) # Store the number of phases in the hash table
        return ht

    dim = ch.simplices[0].shape[0]  # Dimension = Number of components

    if adjacency_tol is None: adjacency_tol = phi_min_spacing * np.sqrt(dim - 1) * 3

    if useHashTable:
        hashtable = make_hashtable(dim)

    nSimplices = ch.simplices.shape[0]
    nPhases = np.empty(nSimplices)

    for i in tqdm(range(1, nSimplices),
                  desc='Counting phases'):  # The first simplex is the unphysical one. This simplex corresponds to the lid in case a ternary system.
        smpl_idx = ch.simplices[i]
        pts_simplex = pts[smpl_idx, :]
        a = construct_adjacency_matrix(pts_simplex, adjacency_tol)

        # THIS IS GENERAL BUT SLOW DUE TO EIGENVALUE COMPUTATION
        d = np.diag(np.sum(a, axis=1))

        if not useHashTable:
            # Graph theory tells you the multiplicity of the eigenvalue 0 of a matrix d-a is equal to the number of stretched edges
            # number of stretched edges correspond to the number of phases.
            d = np.diag(np.sum(a, axis=1))
            eigDict = smat.Matrix(d - a).eigenvals()
            eigDict = {int(key): value for key, value in eigDict.items() if key == 0}
            nPhases[i] = eigDict[0]
        else:
            nPhases[i] = hashtable[a.astype('int32').tobytes()]

    smpl_indices = ch.simplices

    return nPhases, smpl_indices

def delaunayase(hull):
    """
    Perform a Delaunay triangulation on the convex hull
    ... Delaunay triangulation is a triangulation of a set of points
    such that no point is inside the circumcircle of any triangle
    ... Delaunay triangulation is a dual graph of the Voronoi diagram

    Parameters
    ----------
    pts: np.ndarray, ternary coordinates, shape (n, 3)
    hull: scipy.spatial.ConvexHull

    Returns
    -------
    tri: scipy.spatial.Delaunay
    """

    # Perform a Delaunay triangulation on the convex hull
    tri = Delaunay(hull.points[hull.vertices])
    return tri

def scan_phase_space(nComp, chi, n=11, thd=None,):
    """
    Scan a phase space, given a chi matrix.
    ... The chi matrix is the interaction matrix

    Parameters
    ----------
    num_comp: int, number of components
    chi: np.ndarray, chi matrix, shape (num_comp, num_comp)
    n: int, number of grid points along one ternary axis, by default 11
    thd: float, threshold used to determine phases, default: 6/(n-1)
    verbose: bool, whether to print the progress, by default False

    Returns
    -------
    pts: np.ndarray, shape (n, num_comp), volume fractions
    nPhases: np.ndarray, number of phases in each simplex
    indices: np.ndarray, simplices
    """
    chi = np.asarray(chi)
    phi = get_phis(nComp=nComp, n=n) # Shape: (Number of points, nComp)
    dphi = 1 / (n - 1) # resolution of the phi grid

    # GET FLORY-HUGGINS FREE ENERGY
    fe = getF(phi, chi) # Shape: (Number of poitns, )

    # CONVEX-HULL
    hull, pts = convex_hull(phi, fe) # Reminder: `ch` is a ConvexHull class object.

    # COUNT PHASES
    if thd is None:
        thd = 6 * dphi
    nPhases, indices = count_phases(hull, pts, dphi, adjacency_tol=thd, useHashTable=True)

    return pts, nPhases, indices



# PLOTTING TOOLS
###### Helper functions #######
# Ternary system
def tern2xy(phi):
    """
    Convert ternary coordinates to Euclidean coordinates for plotting

    Parameters
    ----------
    phi: np.ndarray, ternary coordinates with shape (n, 3) where n is the number of points

    Returns
    -------
    xy: np.ndarray, Euclidean coordinates with shape (n, 2)
    """
    xy = np.array([(2 / np.sqrt(3)) * (1 - phi[:, 0] / 2 - phi[:, 1]), phi[:, 0]]).T
    xy = xy / (2 / np.sqrt(3))
    return xy

def xy2tern(xy):
    """
    Convert Euclidean coordinates to ternary coordinates

    Parameters
    ----------
    xy: np.ndarray, Euclidean coordinates with shape (n, 2) where n is the number of points

    Returns
    -------
    phi: np.ndarray, ternary coordinates with shape (n, 3)
    """
    x, y = xy[:, 0], xy[:, 1]
    phi3 = x - y / np.sqrt(3)
    phi1 = y / (np.sqrt(3) / 2)
    phi2 = 1 - phi1 - phi3
    phi = np.stack((phi1, phi2, phi3)).T
    return phi



def format_phi_fe(phi, fe):
    """
    Format the ternary coordinates and free energy for convex hull

    Parameters
    ----------
    phi: np.ndarray, ternary coordinates, shape (n, nComp)
    fe: np.ndarray, free energy, shape (n,)

    Returns
    -------
    pts: np.ndarray, coordinates and free energy, shape (n, nComp + 1)
    """
    pts = np.concatenate((phi[:, :-1], fe[:, np.newaxis]), axis=1)
    return pts


# TERNARY PHASE DIAGRAM, NO OF PHASES
def draw_triangle(ax, xy=[[0, 0], [1, 0], [0.5, np.sqrt(3) / 2]], linewidth=1, edgecolor='k',
                  colors=['k', 'r', 'b'], fontsize=10,
                  n=3, offset=0.075):
    """
    Draw a triangle on a ternary plot

    Parameters
    ----------
    ax: mpltern.AxesTernary
    xy: list, vertices of the triangle
    fill: bool, whether to fill the triangle
    linewidth: float, width of the edge
    edgecolor: str, color of the edge

    Returns
    -------
    triangle: mpl.patches.Polygon
    """

    triangle = mpl.patches.Polygon(xy, fill=False, linewidth=linewidth, edgecolor=edgecolor)
    ax.add_patch(triangle)

    textAngle1, textAngle3 = -60, 60
    # offset = np.cos(np.deg2rad(textAngle1)) * 0.15
    print(offset)
    ## phi3
    offset_x, offset_y = - offset / 2, - offset / 1.0
    xmin, xmax = 0 + offset_x, 1 + offset_x
    ymin, ymax = 0 + offset_y, 0 + offset_y
    for x, y, val in zip(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n), np.linspace(0, 1, n)):
        ax.text(x, y, f'{val :.1f}', fontsize=fontsize, rotation=textAngle3,  color=colors[2])

    ## phi1
    offset_x, offset_y = offset / 3, - offset / 5
    xmin, xmax = 1 + offset_x, 0.5 + offset_x*1.1
    ymin, ymax = 0 + offset_y * 0.75, np.sqrt(3) / 2 + offset_y
    for x, y, val in zip(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n), np.linspace(0, 1, n)):
        ax.text(x, y, f'{val :.1f}', fontsize=fontsize,  color=colors[0])

    ## phi2
    offset_x, offset_y = - offset * 1.01, offset / 10
    xmin, xmax = 0.5 + offset_x, 0. + offset_x*1
    ymin, ymax = np.sqrt(3) / 2 + offset_y * 3, 0 + offset_y * 0.8
    for x, y, val in zip(np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n), np.linspace(0, 1, n)):
        ax.text(x, y, f'{val :.1f}', fontsize=fontsize, rotation=textAngle1, color=colors[1])

    # ax.text(0.5 - offset * 1.4, np.sqrt(3) / 2 - offset/2, '0.0', fontsize=fontsize, rotation=textAngle3, color=colors[1])
    # ax.text(0 - offset * 1.1, offset / 10, '1.0', fontsize=fontsize, rotation=textAngle3, color=colors[1])

    # Labels
    ax.text(1.0 + offset , -offset, '$\phi_3$', rotation=textAngle3, color=colors[2])
    ax.text(0.5 - offset * 0.3, np.sqrt(3) / 2 + offset * 1.4, '$\phi_1$', color=colors[0])
    ax.text(- offset * 1.5, -offset, '$\phi_2$', rotation=textAngle1, color=colors[1])

    return triangle

def plot_ternary_phase_diagram(pts, nPhases, indices, cmap='tab10',
                               chi=np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]]), # Title
                               nticks=5, showTieLines=True, nTieLines=30,
                               linewidth=0.7, colorTieLines='gray',
                               fignum=1, subplot=111,
                               cb_kwargs={'fraction': 0.025,'pad': 0.1, 'fontsize': 10},
                               **kwargs,
                               ):
    """
    Plot a ternary phase diagram

    Parameters
    ----------
    pts: np.ndarray, ternary coordinates, shape (n, 3)
    nPhases: np.ndarray, number of phases in each simplex
    indices: np.ndarray, simplices
    cmap: str, colormap
    chi: np.ndarray, interaction matrix, shape (3, 3)

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, matplotlib.axes.Axes
    """
    nComp = 3

    # Convert ternary coordinates to Euclidean coordinates
    xy = tern2xy(pts)
    phi = xy2tern(xy)

    ## PLOTTING
    colors = graph.get_color_from_cmap(cmap=cmap, n=10)[:nComp]
    cmap_discrete, norm = graph.get_discrete_cmap_norm(colors, vmin=1, vmax=nComp)

    fig, ax = graph.set_fig(fignum=fignum, subplot=subplot, projection='ternary',**kwargs)

    # Create a Triangulation object
    tri = mpl.tri.Triangulation(xy[:, 0], xy[:, 1], triangles=indices)
    ax.tripcolor(phi[:, 0], phi[:, 1], phi[:, 2], nPhases,
                 triangles=indices,
                 vmin=1, vmax=nComp,
                 cmap=cmap_discrete,
                 # edgecolors="k",
                 )
    ax.set_aspect(1)

    if showTieLines:
        keep = nPhases == 2
        indices_biphase = indices[keep]
        inc = len(indices_biphase) // nTieLines
        inc = max(1, inc)

        for i, idx_list in enumerate(indices_biphase):
            if i % inc == 0:
                simplex = []
                for idx in idx_list:
                    pts_ = pts[idx]
                    phi_ = pts_
                    phi_[-1] = 1 - np.sum(phi_[:2])
                    simplex.append(phi_)
                simplex = np.asarray(simplex)

                # Plot the tie lines if its node is not on the edge of the triangle
                if not any(np.abs(np.sum(simplex[:, :2], axis=1) - 1) < 1e-4):
                    ax.plot(*simplex[:2, :].T, # simplex[:2, :].T is the first two points of the simplex
                            linewidth=linewidth, color=colorTieLines,
                            alpha=0.5)

        keep = nPhases == 3
        indices_triphase = indices[keep]
        inc = len(indices_triphase) // nTieLines
        inc = max(1, inc)
        for i, idx_list in enumerate(indices_triphase):
            if i % inc == 0:
                simplex = []
                for idx in idx_list:
                    pts_ = pts[idx]
                    phi_ = pts_
                    phi_[-1] = 1 - np.sum(phi_[:2])
                    simplex.append(phi_)
                simplex.append(simplex[0])
                simplex = np.asarray(simplex)
                if not any(np.abs(np.sum(simplex[:, :2], axis=1) - 1) < 1e-4):
                    ax.plot(*simplex.T,
                                linewidth=linewidth, color='lightgreen',
                            alpha=0.5)

    graph.add_discrete_colorbar(ax, colors, vmin=1, vmax=nComp + 1,
                                label='No. of Phases', **cb_kwargs )

    ax.set_tlabel('$\phi_1$')
    ax.set_llabel('$\phi_2$')
    ax.set_rlabel('$\phi_3$')
    fig.tight_layout()
    return fig, ax



# TERNARY PHASE DIAGRAM, FREE ENERGY (FLORY-HUGGINS)
def plot_tern(phi, fe, vmin=None, vmax=None, num=1, cmap='viridis', label='$f/k_BT$'):
    """
    Plot a ternary plot

    Parameters
    ----------
    phi: np.ndarray, ternary coordinates, shape (n, 3)
    fe: np.ndarray, a scalar field such as free energy, shape (n,)
    vmin: float, minimum value of the colorbar
    vmax: float, maximum value of the colorbar
    num: int, figure number
    cmap: str, colormap

    Returns
    -------
    fig, ax: matplotlib.figure.Figure, mpltern.AxesTernary
    """
    # FIGURE
    fig = plt.figure(num=num, figsize=(7, 6))
    fig.subplots_adjust(left=0.075, right=0.85, wspace=0.5)
    axF = fig.add_subplot(111, projection='ternary', rotation=0)
    if vmin is None:
        vmin = np.nanmin(fe)
    if vmax is None:
        vmax = 0

    cs = axF.tripcolor(phi[:, 0], phi[:, 1], phi[:, 2], fe, cmap=cmap, vmin=vmin, vmax=vmax)
    _ = axF.tricontour(phi[:, 0], phi[:, 1], phi[:, 2], fe, levels=5,
                       #                        cmap=cmap,
                       colors='w', linewidths=2,
                       vmin=np.nanmin(fe), vmax=np.nanmax(fe), zorder=10)
    cax = axF.inset_axes([1, 0.48, 0.03, 0.50], transform=axF.transAxes)
    graph.add_colorbar_alone(axF, [vmin, vmax], cax=cax, cmap=cmap, label=label, fontsize=18)

    axF.set_tlabel("$\phi_1$")  # OAW -> AWO   ()
    axF.set_llabel("$\phi_2$")
    axF.set_rlabel("$\phi_3$")
    return fig, axF


###############################

def main():
    pass

if __name__ == 'main':
    main()