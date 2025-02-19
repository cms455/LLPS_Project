'''
Module for plotting figures
'''
import os, copy
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid1 as axes_grid
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
import seaborn as sns
import mpltern # Ternary plots
from cycler import cycler

import itertools
import numpy as np
from fractions import Fraction
from math import modf
import pickle

# Suppress warnings
import warnings

#from tflow.graph import add_colorbar

warnings.filterwarnings("ignore")


# Default variables
# Default color cycle: iterator which gets repeated if all elements were exhausted
#__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__def_colors__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
__color_cycle__ = itertools.cycle(__def_colors__)  #matplotliv v2.0
__old_color_cycle__ = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  #matplotlib classic
cmap = 'magma'

# See all available arguments in matplotlibrc
__fontsize__ = 12
__figsize__ = (3, 3)
params = {'figure.figsize': __figsize__,
          'font.size': __fontsize__,  #text
        'legend.fontsize': 10
          , # legend
         'axes.labelsize': __fontsize__, # axes
         'axes.titlesize': __fontsize__,
          'lines.linewidth': 2,
         'lines.markersize': 3,
         'xtick.labelsize': __fontsize__, # tick
         'ytick.labelsize': __fontsize__,
         'xtick.major.size': 3.5, #3.5 def
         'xtick.minor.size': 2, # 2.
         'ytick.major.size': 3.5,
         'ytick.minor.size': 2,
          'figure.dpi': 100,
         }
default_custom_cycler = {'color': ['r', 'b', 'g', 'y'],
                          'linestyle': ['-', '-', '-', '-'],
                          'linewidth': [3, 3, 3, 3],
                          'marker': ['o', 'o', 'o', 'o'],
                          's': [0,0,0,0]}


## Save a figure
def save(path, ext=['pdf', 'svg', 'png'],
         close=False, verbose=True, fignum=None,
         dpi=150, overwrite=True, tight_layout=False,
         savedata=False, transparent=True, bkgcolor='w',
         **kwargs):
    """
    Save a figure

    Parameters
    ----------
    path: str, path to save a figure
    ext: str, extension of the figure, e.g. 'pdf', 'png', 'svg'
    close: bool, close the figure (default: False)
    verbose: bool, print out the path where the figure is saved
    fignum: int, figure number of the figure to save
    dpi: int, resolution of the figure, default: 150
    overwrite: bool, overwrite if a figure already exists
    tight_layout: bool, use tight_layout
    savedata: bool, save a figure instance, default: True. To load the figure instance,
              try pickle.load(open(path[:-len(ext)-1] + '.pkl', 'rb'))
    transparent: bool, save a figure with a transparent background, default: True
    bkgcolor: str, background color of the figure, default: 'w', `transparent` must be False to activate this option.
    kwargs: keyword arguments to pass to plt.savefig()

    Returns
    -------
    None
    """
    if fignum == None:
        fig = plt.gcf()
    else:
        fig = plt.figure(fignum)
    if dpi is None:
        dpi = fig.dpi

    if tight_layout:
        fig.tight_layout()

    # Separate a directory and a filename from the given path
    directory = os.path.split(path)[0]
    filenames = ["%s.%s" % (os.path.split(path)[1], ext_) for ext_ in ext]

    if directory == '':
        directory = '.'

    # Make sure the directory exists
    directory = Path(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for filename in filenames:
        # The final path to save
        savepath = os.path.join(directory, filename)
        # If the file already exists, generate a new filename
        ver_no = 0
        while os.path.exists(savepath) and not overwrite:
            # this needs to be fixed. right now, it keeps saving to _000.png
            savepath = directory / (os.path.split(path)[1] + '_%03d.' % ver_no + ext)
            ver_no += 1

        # Save the figure
        if transparent:
            plt.savefig(savepath, dpi=dpi, transparent=transparent, **kwargs)
        else:
            plt.savefig(savepath, dpi=dpi, facecolor=bkgcolor, **kwargs)

    # Save a fig instance... This may fail for python2
    if savedata:
        try:
            pickle.dump(fig, open(savepath[:-len(ext)-1] + '.pkl', 'wb'))
        except:
            print('... Unable to pickle a figure instance.')

    # Close a figure
    if close:
        plt.close(fignum)

    if verbose:
        print("Figure saved to", savepath)

## Create a figure and axes
def set_fig(fignum, subplot=111, dpi=100, figsize=None,
            custom_cycler=False, custom_cycler_dict=default_custom_cycler, # advanced features to change a plotting style
            **kwargs):
    """
    Returns Figure and Axes instances
    ... a short sniplet for
        plt.figure(fignum, dpi=dpi, figsize=figsize)
        plt.subplot(subplot, **kwargs)

    Parameters
    ----------
    fignum: int, figure number
    subplot: int, A 3-digit integer. The digits are interpreted as if given separately as three single-digit integers, i.e. fig.add_subplot(235) is the same as fig.add_subplot(2, 3, 5). Note that this can only be used if there are no more than 9 subplots.
    dpi: int,
    figsize: tuple, figure size
    custom_cycler: bool, If True, it enables users to customize a plot style (color cycle, marker cycle, linewidth cycle etc.)
        ... The customized cycler could be passed to custom_cycler_dict.
    custom_cycler_dict: dict, A summary of a plotting style.
        ... E.g.- default_custom_cycler = {'color': ['r', 'b', 'g', 'y'],
                                          'linestyle': ['-', '-', '-', '-'],
                                          'linewidth': [3, 3, 3, 3],
                                          'marker': ['o', 'o', 'o', 'o'],
                                          's': [0,0,0,0]}
        ... The dictionary is turned into a list of cyclers, and passed to ax.set_prop_cycle(custom_cycler).

    kwargs: Visit plt.subplot(**kwargs) for available kwargs

    Returns
    -------
    fig: Figure instance
    ax: Axes instance
    """
    if fignum == -1:
        if figsize is not None:
            fig = plt.figure(dpi=dpi, figsize=figsize)
        else:
            fig = plt.figure(dpi=dpi)
    if fignum == 0:
        fig = plt.cla()  #clear axis
    if fignum > 0:
        if figsize is not None:
            fig = plt.figure(num=fignum, dpi=dpi, figsize=figsize)
            fig.set_size_inches(figsize[0], figsize[1])
        else:
            fig = plt.figure(num=fignum, dpi=dpi)
        fig.set_dpi(dpi)
    if subplot is None:
        subplot = 111
    # >=matplotlib 3.4: fig.add_subplot() ALWAYS creates a new axes instance
    # <matplotlib 3.4: fig.add_subplot() returns an existing Axes instance if it existed
    # ax = fig.add_subplot(subplot, **kwargs, )
    # >matplotlib 3.4 plt.suplot() continues to reuse an existing Axes with a matching subplot spec and equal kwargs.
    ax = plt.subplot(subplot, **kwargs)

    if custom_cycler:
        apply_custom_cyclers(ax, **custom_cycler_dict)

    return fig, ax

def set_size(fig, width_inches, height_inches):
    """
    Set the size of the figure based on the axis box size.

    Parameters
    ----------
    width : float
        Width in inches.
    height : float
        Height in inches.
    fig : matplotlib.figure.Figure
        The figure object to adjust the size for.

    """
    # Draw the figure to get the correct bounding box
    fig.canvas.draw()

    # Get the bounding box of the axis box (the area enclosed by the spines)
    bbox = fig.axes[0].get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    axis_width, axis_height = bbox.width, bbox.height

    # Calculate the scaling factor
    scale_width = width_inches / axis_width
    scale_height = height_inches / axis_height

    # Calculate the new figure size
    fig_width = fig.get_figwidth() * scale_width
    fig_height = fig.get_figheight() * scale_height

    # Set the new figure size
    fig.set_size_inches(fig_width, fig_height)
    return fig

def match_size(ax1, ax2):
    """
    Match the figure size based on two Axes instances. The Axes objects must belong to separate figures.
    Parameters
    ----------
    ax1: Matplotlib.axes.Axes
    ax2: Matplotlib.axes.Axes
    ... This parent figure's size gets adjusted.

    Returns
    -------
    fig1, fig2: figure objects, the canvas size of fig2 gets rescaled to match the given ax sizes
    """

    fig1 = ax1.get_figure()
    fig2 = ax2.get_figure()

    # Draw the figure to get the correct bounding box
    fig1.canvas.draw()
    fig2.canvas.draw()

    # Get the bounding box of the axis box (the area enclosed by the spines)
    bbox1 = ax1.get_window_extent().transformed(fig1.dpi_scale_trans.inverted())
    bbox2 = ax2.get_window_extent().transformed(fig2.dpi_scale_trans.inverted())
    ax1_w, ax1_h = bbox1.width, bbox1.height
    ax2_w, ax2_h = bbox2.width, bbox2.height

    # Calculate the scaling factor
    scale_width = ax1_w / ax2_w
    scale_height = ax1_h / ax2_h

    # Calculate the new figure size
    fig2_width = fig2.get_figwidth() * scale_width
    fig2_height = fig2.get_figheight() * scale_height

    # Set the new figure size
    fig2.set_size_inches(fig2_width, fig2_height)
    return fig1, fig2

def add_subplot_custom(x0, y0 ,x1, y1, fignum=None):
    """
    Create an Axes object at a customized location

    Parameters
    ----------
    x0: left, [0, 1]
    y0: bottom, [0, 1]
    x1: width, [0, 1]
    y1: height, [0, 1]

    Returns
    -------
    ax: Axes object
    """
    fig = plt.figure(num=fignum)
    ax = fig.add_axes([x0, y0, x1-x0, y1-y0])
    return ax

# VISUALIZE MIXTURES

# VISUALIZING FUNCTIONS

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

    add_colorbar_alone(im, ax=ax)
    ax.set_title('$\chi_{ij}$', fontsize=fontsize)
    ax.set_xlabel('Component $j$', fontsize=fontsize*0.9)
    ax.set_ylabel('Component $i$', fontsize=fontsize*0.9)

    ax.set_xticks(np.arange(len(chi))+1)
    ax.set_yticks(np.arange(len(chi))+1)
    ax.set_yticklabels(np.arange(len(chi))[::-1]+1)
    # ax.invert_yaxis()
    return fig, ax

def visualize_mixture(phis, height=3, width=2, fignum=1, subplot=111, ax=None,
                      cmap='turbo', vmin=0.2, vmax=0.9, solvent_color='white', gamma=1.0,
                      lg_kwargs={'fontsize': 12, 'bbox_to_anchor': (1., 1.15),
                                 'columnspacing': 0.7, 'handlelength': 0.5}
                      ):
    """
    Visualizes the volume fractions of all components in a mixture.

    Parameters
    ----------
    phis: 2d array, volume fractions of all components in M partitions (nComp, nPart)
    height: float, height of the figure in inches
    width: float, width of the figure in inches
    fignum: int, figure number
    subplot: int, subplot number
    ax: Axes object, axes to plot the mixture
    cmap: str or colormap object, colormap to use for visualizing the volume fractions
    vmin: float, minimum value of the colormap (choose from 0 to 1)
    vmax: float, maximum value of the colormap (choose from 0 to 1)
    solvent_color: str, color of the solvent, 'powerblue', 'azure', 'white', etc.
    gamma: float, gamma value for color exaggeration
    lg_kwargs: dict, keyword arguments for the legend
        ... bbox_to_anchor: tuple, (x, y), the bbox that the legend is anchored

    Returns
    -------

    """

    def create_n_cmap_bases(n, cmap='turbo', vmin=0., vmax=1., solvent_color='powderblue'):
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

        # #(DEBUGGING)
        # #Show basis colors in a single plot
        # fig, ax = plt.subplots()
        # for i, color in enumerate(basis_colors):
        #     ax.fill_between([0, 1], i, i+1, color=color)
        # ax.axis('off')
        # plt.show()
        # sys.exit()

        # create a linear color map
        colors_list = [get_color_list_gradient(color1=solvent_color, color2=rgb2hex(basis_color), ) for basis_color
                       in basis_colors]
        cmaps = [create_cmap_from_colors(colors) for colors in colors_list]
        return cmaps

    def assign_color(phi, cmap='turbo', solvent_color='powderblue', gamma=1,
                     vmin=0., vmax=1.,):
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
        phi = np.power(phi, gamma) # Exerggerate the color
        ncomp = len(phi)  # number of components
        # cmaps = [truncate_colormap(cmap, minval=i/n, maxval=(i+1)/n) for i in range(n)]
        cmaps = create_n_cmap_bases(ncomp, cmap=cmap, vmin=vmin, vmax=vmax, solvent_color=solvent_color)
        colors = np.empty((4, len(phi)))
        for i in range(ncomp):
            colors[:, i] = cmaps[i](phi[i]) # shape: (4, ncomp)
        # sum colors
        color = np.mean(colors, axis=1) # average over all components
        # color = np.max(colors, axis=1) # average over all components

        # # Optional: Set `1 - solvent concentration` as the alpha- bad idea since it makes the color too transparent
        # color[-1] = 1 - np.sum(phi)
        return color

    ncomps, nparts = phis.shape
    colors = []
    for i in range(nparts):
        phi = phis[:, i] # volume fraction of each partition
        color = assign_color(phi, cmap=cmap, solvent_color=solvent_color, gamma=gamma,
                             vmin=vmin, vmax=vmax)
        colors.append(color)

    # Visualize:
    ## Set up Figure and Axes objects
    if ax is None:
        fig = plt.figure(num=fignum, figsize=(width, height))
        ax = fig.add_subplot(subplot)
    else:
        fig = ax.get_figure()

    for i in range(nparts):
        ax.fill_between([0, width], (height / nparts) * i, (height / nparts) * (i + 1),
                        color=colors[i], linewidth=0)
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    ax.xaxis.set_ticks([])
    ax.yaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    fig.tight_layout()


    # Return color keys for each component
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    basis_colors = [cmap(val) for val in np.linspace(vmin, vmax, ncomps)]  # Color keys for each component

    labels = [f'$\phi_{i}$' for i in range(1, ncomps+1)]


    lg = legend_custom(ax, basis_colors, labels,
                  ncol=min(ncomps+1, 5),
                  frameon=False,
                  **lg_kwargs)


    return fig, ax, basis_colors

def visualize_mixtures(list_of_phis, height=4, width=5, fignum=1,
                       cmap='turbo', vmin=0, vmax=1, solvent_color='powderblue', gamma=1.0,
                       t=None, chi=None, cmap_chi='coolwarm', fontsize=12,
                       lg_kwargs={'fontsize':12, 'bbox_to_anchor': (1., 1.15),
                                  'columnspacing': 0.7, 'handlelength': 0.5}):
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
        if nphis > 1:
            axes_phi = axes
        else:
            axes_phi = [axes]

    for i, (phis, ax) in enumerate(zip(list_of_phis, axes_phi)):
        fig, ax, colors = visualize_mixture(phis, ax=ax,
                                            cmap=cmap, vmin=vmin, vmax=vmax,
                                            solvent_color=solvent_color, gamma=gamma)
    if t is not None:
        dt = t[1] - t[0]
        for t_, ax in zip(t, axes_phi):
            ax.set_title('$t=$' + f'{t_ / dt:.1f}' + '$\Delta t_{\\rm step}$', fontsize=12)

    if chi is not None:
        visualize_chi(chi, ax=axes[0], cmap=cmap_chi, fontsize=fontsize)

    phi_avg_true = np.mean(list_of_phis[0], axis=0) # True average composition
    phi_text = [f'{value:.2f}' for value in phi_avg_true]
    title = ('$\\vec{\phi}=\{ \\phi_{sol}, \phi_i\}=$'
             + f'{phi_text}'.replace('[', '(').replace(']', ')').replace("'", ''))
    # suptitle(title, fontsize=12, y=0.95)

    labels = [f'$\phi_{i}$' for i in range(1, len(phi_avg_true)+1)]
    colors = colors
    legend_custom(axes_phi[0], colors, labels,
                  ncol=min(len(phi_avg_true)+1, 5),
                  frameon=False,
                  **lg_kwargs)

    return fig, axes, colors


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





# Colorbar
class FormatScalarFormatter(mpl.ticker.ScalarFormatter):
    """
    Ad-hoc class to subclass matplotlib.ticker.ScalarFormatter
    in order to alter the number of visible digits on color bars
    """
    def __init__(self, fformat="%03.1f", offset=True, mathText=True):
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
        self.set_scientific(True)
        # Scientific notation is used for data < 10^-n or data >= 10^m, where n and m are the power limits set using set_powerlimits((n,m))
        self.set_powerlimits((0, 0))
    def _set_format(self):
        """
        Call this method to change the format of tick labels

        Returns
        -------

        """

        self.format = self.fformat
        if self._useMathText:
            # self.format = '$%s$' % mpl.ticker._mathdefault(self.format) # matplotlib < 3.1
            self.format = '$%s$' % self.format


    def _update_format(self, fformat):
        self.fformat = fformat
        self._set_format()

def reset_sfmt(fformat="%03.1f"):
    """
    Reset the format of the scalar formatter

    Parameters
    ----------
    fformat: str, format of the scalar formatter

    Returns
    -------
    None
    """
    global sfmt
    sfmt = FormatScalarFormatter() # Default format: "%04.1f"
    # sfmt.fformat = fformat # update format
    # sfmt._set_format() # this updates format for scientific nota
    sfmt._update_format(fformat)


reset_sfmt() # Initialize the scalar formatter
def get_sfmt(fformat="%03.1f"):
    """
    Get the scalar formatter

    Parameters
    ----------
    fformat: str, format of the scalar formatter

    Returns
    -------
    sfmt: FormatScalarFormatter instance
    """
    global sfmt
    return sfmt


def add_colorbar_alone(ax, values, cax=None, cmap=cmap, label=None, fontsize=None, option='normal', fformat=None,
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, location='right', color='k',
                 size='5%', pad=0.15, **kwargs):
    """
    Add a colorbar to a figure without a mappable
    ... It creates a dummy mappable with given values

    ... LOCATION OF CAX
    fig, ax = set_fig(1, 111)
    w, pad, size = 0.1, 0.05, 0.05
    add_colorbar_alone(ax, [0, 1], pad=float2pc(pad), size=float2pc(size), tight_layout=False)
    add_subplot_axes(ax, [1-w-(1-1/(1.+pad+size)), 0.8, w, 0.2])


    Parameters
    ----------
    ax: Axes instance
    values: 1D array-like- min and max values of values are found from this array
    cmap: str, cmap instance
    label: str, label of the color bar
    fontsize: float, fontsize of the label
    option: str, choose from 'normal' and 'scientific'
    ... if 'scientific', the color bar is shown in a scientific format like 1x10^exponent
    fformat: str, default: None equivalent to "%03.1f"
    tight_layout: bool, if True, fig.tight_layout() is called.
    ticklabelsize: float
    ticklabel: 1d array-like
    aspect:
    ...  Adding a color bar may distort the aspect ratio. Fix it.
    if aspect == 'equal':
        ax.set_aspect('equal')
    location
    color
    kwargs

    Returns
    -------
    cb:
    """
    fig = ax.get_figure()

    # number of values
    n = np.asarray(values).size
    # get min/max values
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    # vmin, vmax = 0, len(values)


    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if cax is None:
        # make an axis instance for a colorbar
        ## divider.append_axes(location, size=size, pad=pad) creates an Axes
        ## s.t. the size of the cax becomes 'size' (e.g.'5%') of the ax.
        divider = axes_grid.make_axes_locatable(ax)
        cax = divider.append_axes(location, size=size, pad=pad)


    if option == 'scientific':
        if fformat is not None:
            global sfmt
            sfmt.fformat = fformat

        cb = fig.colorbar(sm, cax=cax, format=sfmt, **kwargs)
        reset_sfmt()
    else:
        cb = fig.colorbar(sm, cax=cax,  **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if label is not None:
        if fontsize is None:
            cb.set_label(label, color=color)
        else:
            cb.set_label(label, fontsize=fontsize, color=color)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect == 'equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()
    return cb

def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                 tight_layout=True, ticklabelsize=None, ticklabel=None,
                 aspect = None, useMiddle4Ticks=False, **kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    if not useMiddle4Ticks:
        vmin, vmax = vmin -  tick_spacing / 2., vmax -  tick_spacing / 2.
    ticks = np.linspace(vmin, vmax, len(colors) + 1) + tick_spacing / 2.  # tick positions

    # if there are too many ticks, just use 3 ticks
    if len(ticks) > 10:
        n = len(ticks)
        ticks = [ticks[0], ticks[n//2]-1, ticks[-2]]
        if ticklabel is not None:
            ticklabel = [ticklabel[0], ticklabel[n/2], ticklabel[-1]]


    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if option == 'scientific':
        cb = fig.colorbar(sm, ticks=ticks, format=sfmt, ax=ax, **kwargs)
    else:
        cb = fig.colorbar(sm, ticks=ticks, ax=ax, **kwargs)

    if ticklabel is not None:
        cb.ax.set_yticklabels(ticklabel)

    if not label is None:
        if fontsize is None:
            cb.set_label(label)
        else:
            cb.set_label(label, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)

    # Adding a color bar may distort the aspect ratio. Fix it.
    if aspect=='equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb

### Axes
## Set axes to semilog or loglog
def tosemilogx(ax=None, xmin_default=1e-2, **kwargs):
    if ax == None:
        ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    if xmin == 0:
        ax.set_xlim(xmin_default, xmax)
    ax.set_xscale("log", **kwargs)
def tosemilogy(ax=None, ymin_default=1e-6, **kwargs):
    if ax == None:
        ax = plt.gca()
    ymin, ymax = ax.get_ylim()
    if ymin == 0:
        ax.set_ylim(ymin_default, ymax)
    ax.set_yscale("log", **kwargs)
def tologlog(ax=None, **kwargs):
    if ax == None:
        ax = plt.gca()
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmin==0:
        ax.set_xlim(xmin, xmax)
    if ymin==0:
        ax.set_ylim(ymin, ymax)
    ax.set_xscale("log", **kwargs)
    ax.set_yscale("log", **kwargs)

# Ticks
def set_xtick_interval(ax, tickint):
    """
    Sets x-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tickint))

def set_ytick_interval(ax, tickint):
    """
    Sets y-tick interval as tickint
    Parameters
    ----------
    ax: Axes object
    tickint: float, tick interval

    Returns
    -------

    """
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tickint))

def force2showLogMinorTicks(ax, subs='all', numticks=9, axis='both'):
    """
    Force to show the minor ticks in the logarithmic axes
    ... the minor ticks could be suppressed due to the limited space
    Parameters
    ----------
    ax: Axes instance
    subs: str, list, or np.array, 'all' is equivalent to np.arange(1, 10)
    numticks: int, make this integer high to show the minor ticks

    Returns
    -------

    """
    if axis in ['x', 'both']:
        ax.xaxis.set_minor_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position
    if axis in ['y', 'both']:
        ax.yaxis.set_minor_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position

def force2showLogMajorTicks(ax, subs=[1.], numticks=9, axis='both'):
    """
    Force to show the minor ticks in the logarithmic axes
    ... the minor ticks could be suppressed due to the limited space
    Parameters
    ----------
    ax: Axes instance
    subs: str, list, or np.array, 'all' is equivalent to np.arange(1, 10)
    ... to
    numticks: int, make this integer high to show the minor ticks

    Returns
    -------

    """
    if axis in ['x', 'both']:
        ax.xaxis.set_major_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position
    if axis in ['y', 'both']:
        ax.yaxis.set_major_locator(ticker.LogLocator(subs=subs, numticks=numticks))  # set the ticks position

def hide_xticks(ax):
    """
    Hides x-ticks and their labels

    Parameters
    ----------
    ax

    Returns
    -------

    """
    for tick in ax.xaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

def hide_yticks(ax):
    """
    Hides y-ticks and their labels

    Parameters
    ----------
    ax

    Returns
    -------

    """
    for tick in ax.yaxis.get_major_ticks():
        tick.tick1line.set_visible(False)
        tick.tick2line.set_visible(False)
        tick.label1.set_visible(False)
        tick.label2.set_visible(False)

##Title
def title(ax, title, **kwargs):
    """
    ax.set_title(title, **kwargs)
    ... if you want more space for the tile, try "pad=50"

    Parameters
    ----------
    ax
    title
    subplot
    kwargs

    Returns
    -------

    """
    ax.set_title(title, **kwargs)

def suptitle(title, fignum=None,
             tight_layout=True,
             rect=[0, 0.03, 1, 1.3],
             **kwargs):
    """
    Add a centered title to the figure.
    If fignum is given, it adds a title, then it reselects the figure which selected before this method was called.
    ... this is because figure class does not have a suptitle method.
    ...
    Parameters
    ----------
    title
    fignum
    kwargs

    Returns
    -------

    """
    if fignum is not None:
        plt.figure(fignum)
    fig = plt.gcf()

    plt.suptitle(title, **kwargs)
    if tight_layout:
        fig.tight_layout(rect=rect)

##Text
def draw_power_triangle(ax, x, y, exponent, w=None, h=None, facecolor='none', edgecolor='r', alpha=1.0, flip=False,
                        linewidth=1.,
                        fontsize=__fontsize__, set_base_label_one=False, beta=20, zorder=100,
                        x_base=None, y_base=None, x_height=None, y_height=None,
                        **kwargs):
    """
    Draws a triangle which indicates a power law in the log-log plot.

    Parameters
    ----------
    ax: matplotlib.axes._subplots.AxesSubplot object
        ... get it like plt.gca()
    x: float / int
        ... x coordinate of the triangle drawn on the plot
    y: float / int
        ... x coordinate of the triangle drawn on the plot
    exponent: float / int
        ... exponent of the power law
        ... Y = X^exponent
    w: float / int
        ... number of decades for the drawn triangle to span on the plot
        ... By default, this function draws a triangle with size of 0.4 times the width of the plot
    h: float / int
        ... number of decades for the drawn triangle to span on the plot
    facecolor: str
        ... face color of the drawn triangle, default: 'none' (transparent)
        ... passed to mpatches.PathPatch object
    edgecolor: str
        ... edge color of the drawn triangle, default: 'r'
        ... passed to mpatches.PathPatch object
    alpha: float [0, 1]
        ... alpha value of the drawn triangle
    flip: bool
        ... If True, it will flip the triangle horizontally.
    fontsize: float / int
        ... fontsize of the texts to indicate the exponent aside the triangle
    set_base_label_one: bool, default: False
        ... If True, it will always annotate the base as '1' and alter the text for the height accordingly.
        ... By default, it will annotate the base and the height using the closest integer pair.
    beta: float / int, default: 20
        ... This is used to control the spacing between the text and the drawn triangle
        ... The higher beta is, the less spacing between the text and the triangle
    zorder: zorder of triangle, default: 0
    kwargs: the other kwargs will be passed to ax.text()

    Returns
    -------

    """

    def simplest_fraction_in_interval(x, y):
        """Return the fraction with the lowest denominator in [x,y]."""
        if x == y:
            # The algorithm will not terminate if x and y are equal.
            raise ValueError("Equal arguments.")
        elif x < 0 and y < 0:
            # Handle negative arguments by solving positive case and negating.
            return -simplest_fraction_in_interval(-y, -x)
        elif x <= 0 or y <= 0:
            # One argument is 0, or arguments are on opposite sides of 0, so
            # the simplest fraction in interval is 0 exactly.
            return Fraction(0)
        else:
            # Remainder and Coefficient of continued fractions for x and y.
            xr, xc = modf(1 / x);
            yr, yc = modf(1 / y);
            if xc < yc:
                return Fraction(1, int(xc) + 1)
            elif yc < xc:
                return Fraction(1, int(yc) + 1)
            else:
                return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

    def approximate_fraction(x, e):
        """Return the fraction with the lowest denominator that differs
        from x by no more than e."""
        return simplest_fraction_in_interval(x - e, x + e)

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    if xmin < 0:
        xmin = 1e-16
    if ymin < 0:
        ymin = 1e-16
    exp_xmax, exp_xmin = np.log10(xmax), np.log10(xmin)
    exp_ymax, exp_ymin = np.log10(ymax), np.log10(ymin)
    exp_x, exp_y = np.log10(x), np.log10(y)

    # Default size of the triangle is 0.4 times the width of the plot
    if w is None and h is None:
        exp_w = (exp_xmax - exp_xmin) * 0.4
        exp_h = exp_w * exponent
    elif w is None and h is not None:
        exp_h = h
        exp_w = exp_h / exponent
    elif w is not None and h is None:
        exp_w = w
        exp_h = exp_w * exponent
    else:
        exp_w = w
        exp_h = h

    w = 10 ** (exp_x + exp_w) - 10 ** exp_x  # base of the triangle
    h = 10 ** (exp_y + exp_h) - 10 ** exp_y  # height of the triangle
    if not flip:
        path = mpl.path.Path([[x, y], [x + w, y], [x + w, y + h], [x, y]])
    else:
        path = mpl.path.Path([[x, y], [x, y + h], [x + w, y + h], [x, y]])
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder, linewidth=linewidth)
    ax.add_patch(patch)

    # annotate
    # beta = 20. # greater beta corresponds to less spacing between the texts and the triangle edges
    if any([item is None for item in [x_base, y_base, x_height, y_height]]):
        if exponent >= 0 and not flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y - (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_w + exp_x + 0.4 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        elif exponent < 0 and not flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + 0.3 * (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_w + exp_x + 0.4 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        elif exponent >= 0 and flip:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.4), 10 ** (exp_y + exp_h + 0.3 * (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_x - (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.5)
        else:
            x_base, y_base = 10 ** (exp_x + exp_w * 0.5), 10 ** (exp_y + exp_h - (exp_ymax - exp_ymin) / beta)
            x_height, y_height = 10 ** (exp_x - 0.6 * (exp_xmax - exp_xmin) / beta), 10 ** (exp_y + exp_h * 0.6)

    if set_base_label_one:
        ax.text(x_base, y_base, '1', fontsize=fontsize)
        ax.text(x_height, y_height, '%.2f' % exponent, fontsize=fontsize)
    else:
        # get the numbers to put on the graph to indicate the power
        exponent_rational = approximate_fraction(exponent, 0.0001)
        ax.text(x_base, y_base, str(np.abs(exponent_rational.denominator)), fontsize=fontsize, **kwargs)
        ax.text(x_height, y_height, str(np.abs(exponent_rational.numerator)), fontsize=fontsize, **kwargs)


##Clear a canvas
def clf(fignum=None):
    plt.figure(fignum)
    plt.clf()
def close(*argv, **kwargs):
    plt.close(*argv, **kwargs)

## Color cycle
def skipcolor(numskip, color_cycle=__color_cycle__):
    """ Skips numskip times in the color_cycle iterator
        Can be used to reset the color_cycle"""
    for i in range(numskip):
        next(color_cycle)
def countcolorcycle(color_cycle = __color_cycle__):
    return sum(1 for color in color_cycle)

def get_default_color_cycle():
    return __color_cycle__

def get_first_n_colors_from_color_cycle(n):
    color_list = []
    for i in range(n):
        color_list.append(next(__color_cycle__))
    return color_list

def get_first_n_default_colors(n):
    return __def_colors__[:n]


def apply_custom_cyclers(ax, color=['r', 'b', 'g', 'y'], linestyle=['-', '-', '-', '-'], linewidth=[3, 3, 3, 3],
                         marker=['o', 'o', 'o', 'o'], s=[0,0,0,0], **kwargs):

    """
    This is a simple example to apply a custom cyclers for particular plots.
    ... This simply updates the rcParams so one must call this function BEFORE ceration of the plots.
    ... e.g.
            fig, ax = set_fig(1, 111)
            apply_custom_cyclers(ax, color=['r', 'b', 'g', 'y'])
            ax.plot(x1, y1)
            ax.plot(x2, y2)
            ...

    Parameters
    ----------
    ax: mpl.axes.Axes instance
    color: list of strings, color
    linewidths: list of float values, linewidth
    linestyles: list of strings, linestyle
    marker: list of strings, marker
    s: list of float values, marker size

    Returns
    -------
    None

    """
    custom_cycler = cycler(color=color) + cycler(linestyle=linestyle) + cycler(lw=linewidth) + cycler(marker=marker) + cycler(markersize=s)
    ax.set_prop_cycle(custom_cycler)


def create_cmap_using_values(colors=None, color1='greenyellow', color2='darkgreen', color3=None, n=100):
    """
    Create a colormap instance from a list
    ... This returns mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    ... One can specify the colors by giving two or three color names recognized by matplotlib.
    ...... e.g.- Passing color1='red' and color2='blue' returns a colormap that linearly interpolates from red to blue.

    Parameters
    ----------
    colors: list of colors, [(0-255, 0-255, 0-255), ..., (0-255, 0-255, 0-255)]
    color1: str, color name ('red', 'blue', etc.)
    color2: str, color name ('red', 'blue', etc.)
    n: int, number of points to interpolate colors when color1 and color2 are given

    Returns
    -------
    cmap: cmap object
    """
    if colors is None:
        colors = get_color_list_gradient(color1=color1, color2=color2, color3=color3, n=n)
    cmap_name = 'new_cmap'
    newcmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    return newcmap


def create_cmap_from_colors(colors_list, name='newmap'):
    """
    Returns mpl.colors.LinearSegmentedColormap.from_list(name, colors_list)

    Parameters
    ----------
    colors_list
    name

    Returns
    -------

    """
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(name, colors_list)


def get_colors_and_cmap_using_values(values, cmap=None, color1='greenyellow', color2='darkgreen', color3=None,
                                     vmin=None, vmax=None, n=100):
    """
    Returns colors (list), cmap instance, mpl.colors.Normalize instance assigned by the
    ...

    Parameters
    ----------
    values: 1d array-like,
    cmap: str or  matplotlib.colors.Colormap instance
    color1: str, color name or hex code
    color2: str, color name or hex code
    vmin: float, default: None
    vmax: float, default: None
    n: int, default: 100

    Returns
    -------
    colors: list, list of colors
    cmap: matplotlib.colors.Colormap instance
    norm: matplotlib.colors.Normalize instance
    """
    values = np.asarray(values)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    if cmap is None:
        cmap = create_cmap_using_values(color1=color1, color2=color2, color3=color3, n=n)
    else:
        cmap = plt.get_cmap(cmap, n)
    # normalize
    # vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap(norm(values))
    return colors, cmap, norm

def get_norm(cmapRange):
    """
    Returns a Normalize instance for the given range

    Parameters
    ----------
    cmapRange: 1d array-like, range of the colormap

    Returns
    -------
    norm: matplotlib.colors.Normalize instance
    """
    vmin, vmax = np.nanmin(cmapRange), np.nanmax(cmapRange)
    norm = plt.Normalize(vmin, vmax)
    return norm


def get_linear_cmap(color1='greenyellow', color2='darkgreen', color3=None, n=100):
    """
    Returns a linear colormap instance
    ... color1---color2---color3
    Parameters
    ----------
    color1: str, color name or hex code
    color2: str, color name or hex code
    color3: str, color name or hex code
    n: int, number of colors

    Returns
    -------
    cmap: matplotlib.colors.Colormap instance
    """
    colors, cmap = get_color_list_gradient(color1=color1, color2=color2, color3=color3, n=n, return_cmap=True)
    return cmap

def get_color_list_gradient(color1='greenyellow', color2='darkgreen', color3=None, n=100, return_cmap=False):
    """
    Returns a list of colors in RGB between color1 and color2
    Input (color1 and color2) can be RGB or color names set by matplotlib
    ... color1-color2-color3

    Parameters
    ----------
    color1
    color2
    n: length of the returning list

    Returns
    -------
    color_list
    """
    if color3 is None:
        # convert color names to rgb if rgb is not given as arguments
        if not color1[0] == '#':
            color1 = cname2hex(color1)
        if not color2[0] == '#':
            color2 = cname2hex(color2)
        color1_rgb = hex2rgb(color1)  # np array
        color2_rgb = hex2rgb(color2)  # np array

        r = np.linspace(color1_rgb[0], color2_rgb[0], n)
        g = np.linspace(color1_rgb[1], color2_rgb[1], n)
        b = np.linspace(color1_rgb[2], color2_rgb[2], n)
        color_list = list(zip(r, g, b))
    else:
        # convert color names to rgb if rgb is not given as arguments
        if not color1[0] == '#':
            color1 = cname2hex(color1)
        if not color2[0] == '#':
            color2 = cname2hex(color2)
        if not color3[0] == '#':
            color3 = cname2hex(color3)
        color1_rgb = hex2rgb(color1)   # np array
        color2_rgb = hex2rgb(color2) # np array
        color3_rgb = hex2rgb(color3)   # np array

        n_middle = int((n-1)/2)

        r1 = np.linspace(color1_rgb[0], color2_rgb[0], n_middle, endpoint=False)
        g1 = np.linspace(color1_rgb[1], color2_rgb[1], n_middle, endpoint=False)
        b1 = np.linspace(color1_rgb[2], color2_rgb[2], n_middle, endpoint=False)
        color_list1 = list(zip(r1, g1, b1))

        r2 = np.linspace(color2_rgb[0], color3_rgb[0], n-n_middle)
        g2 = np.linspace(color2_rgb[1], color3_rgb[1], n-n_middle)
        b2 = np.linspace(color2_rgb[2], color3_rgb[2], n-n_middle)
        color_list2 = list(zip(r2, g2, b2))
        color_list = color_list1 + color_list2
    if return_cmap:
        cmap = create_cmap_using_values(colors=color_list, n=n)
        return color_list, cmap
    else:
        return color_list

def get_color_from_cmap(cmap='viridis', n=10, lut=None, reverse=False):
    """
    A simple function which returns a list of RGBA values from a cmap (evenly spaced)
    ... If one desires to assign a color based on values, use get_colors_and_cmap_using_values()
    ... If one prefers to get colors between two colors of choice, use get_color_list_gradient()
    Parameters
    ----------
    cmapname: str, standard cmap name
    n: int, number of colors
    lut, int,
        ... If lut is not None it must be an integer giving the number of entries desired in the lookup table,
        and name must be a standard mpl colormap name.

    Returns
    -------
    colors

    """
    cmap = mpl.cm.get_cmap(cmap, lut)
    if reverse:
        cmap = cmap.reversed()
    colors = cmap(np.linspace(0, 1, n, endpoint=True))
    return colors

def create_weight_shifted_cmap(cmapname, ratio=0.75, vmin=None, vmax=None, vcenter=None, n=500):
    """
    Creates a cmap instance of a weight-shifted colormap

    Parameters
    ----------
    cmapname: str
    ratio
    vmin
    vmax
    vcenter
    n

    Returns
    -------

    """
    if vmin is not None and vmax is not None and vcenter is not None:
        if vmax <= vmin:
            raise ValueError('... vmax must be greater than vmin')
        if vcenter <= vmin or vcenter >= vmax:
            raise ValueError('vcenter must take a value between vmin and vmax')
        vrange = vmax - vmin
        ratio = (vcenter - vmin) / vrange

    cmap_ = mpl.cm.get_cmap(cmapname, n)
    colorNeg = cmap_(np.linspace(0, 0.5, int(n * ratio)))
    colorPos = cmap_(np.linspace(0.5, 1, n - int(n * ratio)))
    newcolors = np.concatenate((colorNeg, colorPos), axis=0)
    newcmap = mpl.colors.ListedColormap(newcolors, name='shifted_' + cmapname)  # custom cmap
    return newcmap


def get_discrete_cmap_norm(colors, vmin=0, vmax=None):
    """
    Return a colormap and a norm for a discrete colorbar

    Parameters
    ----------
    colors: list of colors
    vmin: int, default: 0
    vmax: int, default: None

    Returns
    -------
    cmap, norm: colormap, norm
    """
    if vmax is None:
        vmax = len(colors)

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    return cmap, norm


# Scale bar
class AnchoredScaleBar(AnchoredOffsetbox):
    def __init__(self, transform, sizex=0, sizey=0, labelx=None, labely=None,
                 loc=4,
                 xmin=None, ymin=None, width=None, height=None,
                 pad=0.1, borderpad=0.1, sep=2, prop=None, color="black", barwidth=None,
                 fontsize=None, frameon=False,
                 **kwargs):
        """
        Draw a horizontal and/or vertical  bar with the size in data coordinate
        of the give axes. A label will be drawn underneath (center-aligned).
        - transform : the coordinate frame (typically axes.transData)
        - sizex,sizey : width of x,y bar, in data units. 0 to omit
        - labelx,labely : labels for x,y bars; None to omit
        - loc : position in containing axes
        - pad, borderpad : padding, in fraction of the legend font size (or prop)
        - sep : separation between labels and bars in points.
        - **kwargs : additional arguments passed to base class constructor
        """
        from matplotlib.patches import Rectangle
        from matplotlib.offsetbox import AuxTransformBox, VPacker, HPacker, TextArea, DrawingArea
        bars = AuxTransformBox(transform)
        if sizex:
            bars.add_artist(Rectangle((0, 0), sizex, 0, ec=color, lw=barwidth, fc="none"))
        if sizey:
            bars.add_artist(Rectangle((0, 0), 0, sizey, ec=color, lw=barwidth, fc="none"))

        if sizex and labelx:
            self.xlabel = TextArea(labelx, textprops={'size': fontsize, 'color': color})
            bars = VPacker(children=[bars, self.xlabel], align="center", pad=0, sep=sep)
        if sizey and labely:
            self.ylabel = TextArea(labely, textprops={'size': fontsize, 'color': color})
            bars = HPacker(children=[self.ylabel, bars], align="center", pad=0, sep=sep)
        AnchoredOffsetbox.__init__(self, loc, pad=pad, borderpad=borderpad,
                                   child=bars, prop=prop, frameon=frameon,
                                   #                                    bbox_to_anchor = kwargs['bbox_to_anchor'], #(xmin, ymin, width, height)
                                   #                                    bbox_transform =  kwargs['bbox_transform'],
                                   **kwargs,
                                   )

def add_scalebar(ax,
                 show_xscale=True, show_yscale=False,
                 xscale=None, yscale=None, units='mm',
                 labelx=None, labely=None,
                 barwidth=2,
                 loc='lower right',
                 facecolor=None,
                 hidex=True, hidey=True,
                 **kwargs):
    """ Add scalebars to axes
    Adds a set of scale bars to *ax*, matching the size to the ticks of the plot
    and optionally hiding the x and y axes
    - ax : the axis to attach ticks to
    - show_xscale, show_yscale: bool, if True, it draws a scale bar
    - xscale, yscale: float, length of the scale bar(s)
    - labelx, labely: str, label of x and y
    - units: if labelx or labely is not provided, it creates a label such that "50.0" + 'units'
    - barwidth: float, linewidth of a scale bar
    - loc: str, int, or tuple, location of the scale bar
        ... If loc is tuple, loc = (xmin, ymin, width, height) specifies the location of a scale bar.
        ... xmin, ymin, width, height must be between 0-1
        ... To adjust the label position, vary 'sep'.
    - hidex,hidey : if True, hide x-axis and y-axis of the parent Axes instance
    - **kwargs : additional arguments passed to AnchoredScaleBars
    Returns a created scalebar object

    Example: Place a scale bar outside the axis
        add_scalebar(ax, color='w', xscale=30, loc=(0.78, 0.67, 0.05, 0.05),
    #              barwidth=2, pad=-0.2, sep=-12)


    """

    def f(axis):
        l = axis.get_majorticklocs()
        return len(l) > 1 and (l[1] - l[0])

    # X SCALE
    if show_xscale:
        if xscale is None:
            kwargs['sizex'] = f(ax.xaxis)
        else:
            kwargs['sizex'] = xscale

        if labelx is None:
            kwargs['labelx'] = str(kwargs['sizex']) + units
        else:
            kwargs['labelx'] = labelx
    if show_yscale:
        if yscale is None:
            kwargs['sizey'] = f(ax.yaxis)
        else:
            kwargs['sizey'] = yscale

        if labely is None:
            kwargs['labely'] = str(kwargs['sizey']) + units
        else:
            kwargs['labely'] = labely
    kwargs['barwidth'] = barwidth
    if isinstance(loc, list) or isinstance(loc, tuple):
        xmin, ymin, width, height = loc
        if all(loc):
            kwargs['bbox_to_anchor'] = Bbox.from_bounds(xmin, ymin, width, height)  # (xmin, ymin, width, height)
            kwargs['bbox_transform'] = ax.figure.transFigure
            loc = 1
    sb = AnchoredScaleBar(ax.transData, loc=loc, **kwargs)
    if facecolor is not None:
        sb.patch.set_linewidth(0)
        sb.patch.set_facecolor(facecolor)
    ax.add_artist(sb)

    # Hide x-, y-axis of the axis
    if hidex: ax.xaxis.set_visible(False)
    if hidey: ax.yaxis.set_visible(False)
    if hidex and hidey: ax.set_frame_on(False)

    return sb


def choose_colors(**kwargs):
    """
    Equivalent of sns.choose_cubehelix_palette()

    Example: COLOR CURVES BASED ON A QUANTITY 'Z'
        # What is Z?
        z = [0, 0.25, 0.5, 0.75, 1]
        # Choose colors
        colors = choose_colors()
        # Set the colors as a default color cycle
        set_default_color_cycle(n=len(z), colors=colors)
        # Plot your data...
        plot(x1, y1) # color1
        plot(x2, y2) # color2
        ...
        # Add a colorbar (On the same figure)
        add_colorbar_alone(plt.gca(), z, colors=colors) # plot a stand-alone colorbar in the figure
        # Add a colorbar (On the different figure)
        plot_colorbar(z, colors=colors, fignum=2) # plot a stand-alone colorbar on a new figure

    Parameters
    ----------
    kwargs

    Returns
    -------
    colors
    """
    colors = sns.choose_cubehelix_palette(**kwargs)
    return colors


def hex2rgb(hex, normalize=True):
    """
    Converts a HEX code to RGB in a numpy array
    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.strip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    if normalize:
        rgb = rgb / 255.

    return rgb

def cname2hex(cname):
    """
    Converts a color registered on matplotlib to a HEX code
    Parameters
    ----------
    cname

    Returns
    -------

    """
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS) # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except NameError:
        print(cname, ' is not registered as default colors by matplotlib!')
        return None

def cname2rgba(cname, normalize=True):
    hex = cname2hex(cname)
    rgba = hex2rgb(hex)
    if normalize:
        rgba = rgba / 255
    return rgba

def set_default_color_cycle(name='tab10', n=10, colors=None, reverse=False):
    """
    Sets a color cycle for plotting

    sns_palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] # sns_palettes
    matplotlab cmap names: 'tab10' (default cmap of mpl), 'tab20', 'Set1', 'Set2' etc.
    (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    ... One may specify the color cycles using the existing color maps (seaborn and matplotlib presets)
        or a list of colors specified by a user.
    ... For the presets, pass a name of the colormap like "tab10" (mpl default), "muted" (seaborn defualt)
    ... For a more customized color cycle, pass a list of colors to 'colors'.

    Parameters
    ----------
    name: str, name of the cmap
    n: int, number of colors
    colors: list, a list of colors like ['r', 'b', 'g', 'magenta']

    Returns
    -------
    None
    """
    if colors is None:
        colors = sns.color_palette(name, n_colors=n)
        if reverse:
            colors.reverse()
    sns.set_palette(colors)
    return colors

def set_color_cycle(cmapname='tab10', ax=None, n=10, colors=None):
    """
    Sets a color cycle of a particular Axes instance

    sns_palettes = ['deep', 'muted', 'bright', 'pastel', 'dark', 'colorblind'] # sns_palettes
    matplotlab cmap names: 'tab10' (default cmap of mpl), 'tab20', 'Set1', 'Set2' etc.
    (https://matplotlib.org/stable/tutorials/colors/colormaps.html)
    ... One may specify the color cycles using the existing color maps (seaborn and matplotlib presets)
        or a list of colors specified by a user.
    ... For the presets, pass a name of the colormap like "tab10" (mpl default), "muted" (seaborn defualt)
    ... For a more customized color cycle, pass a list of colors to 'colors'.

    Parameters
    ----------
    cmapname: str, name of the cmap like 'viridis', 'jet', etc.
    n: int, number of colors
    colors: list, a list of colors like ['r', 'b', 'g', 'magenta']

    Returns
    -------
    None
    """
    if colors is None:
        colors = sns.color_palette(cmapname, n_colors=n)
    if ax is None:
        sns.set_palette(colors)
    else:
        ax.set_prop_cycle(color=colors)

def set_color_cycle_custom(ax, colors=__def_colors__):
    """
    Sets a color cycle using a list
    Parameters
    ----------
    ax
    colors: list of colors in rgb/cnames/hex codes

    Returns
    -------

    """
    ax.set_prop_cycle(color=colors)

def set_color_cycle_gradient(ax, color1='greenyellow', color2='navy', n=10):
    colors = get_color_list_gradient(color1, color2, n=n)
    ax.set_prop_cycle(color=colors)


# Figure settings
def update_figure_params(params):
    """
    update a default matplotlib setting
    e.g. params = { 'legend.fontsize': 'x-large',
                    'figure.figsize': (15, 5),
                    'axes.labelsize': 'x-large',
                    'axes.titlesize':'x-large',
                    'xtick.labelsize':'x-large',
                    'ytick.labelsize':'x-large'}
    ... pylab.rcParams.update(params)
    Parameters
    ----------
    params: dictionary

    Returns
    -------
    None
    """

    plt.rcParams.update(params)

def reset_figure_params():
    plt.rcParams.update(plt.rcParamsDefault)


# Use the deafult plotting settings
# reset_figure_params()
# Use my custom settings
update_figure_params(params)

## 3D plotting
def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])



# Plotting styles
def show_plot_styles():
    """Prints available plotting styles"""
    style_list = ['default'] + sorted(style for style in plt.style.available)
    print(style_list)
    return style_list

def use_plot_style(stylename):
    """Reminder for me how to set a plotting style"""
    plt.style.use(stylename)

def set_plot_style(stylename):
    """Reminder for me how to set a plotting style"""
    plt.style.use(stylename)

def show_availbale_markerstyles():
    """Returns a list of available markers for ax.scatter()"""
    filled_markers = list(Line2D.filled_markers)
    unfilled_markers = [m for m, func in Line2D.markers.items()
               if func != 'nothing' and m not in Line2D.filled_markers]
    markers = filled_markers + unfilled_markers
    return markers

# Embedded plots
def add_subplot_axes(ax, rect, axisbg='w', alpha=1, spines2hide=['right', 'top'], **kwargs):
    """
    Creates a sub-subplot inside the subplot (ax)
    rect: list, [x, y, width, height] e.g. rect = [0.2,0.2,0.7,0.7]

    Parameters
    ----------
    ax
    rect: list, [x, y, width, height]  e.g. rect = [0.2,0.2,0.7,0.7]
    axisbg: background color of the newly created axes object

    Returns
    -------
    subax, Axes class object
    """

    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x, y, width, height], **kwargs)
    subax.set_facecolor(axisbg)
    subax.patch.set_alpha(alpha)
    x_labelsize = subax.get_xticklabels()[0].get_size()
    y_labelsize = subax.get_yticklabels()[0].get_size()
    x_labelsize *= rect[2]**0.3
    y_labelsize *= rect[3]**0.3
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    if spines2hide is not None:
        subax.spines[spines2hide].set_visible(False)
    return subax

def create_inset_ax(ax, rect, axisbg='w', alpha=1, **kwargs):
    """DEPRICATED. Use add_subplot_axes"""
    return add_subplot_axes(ax, rect, axisbg=axisbg, alpha=alpha, **kwargs)

## misc.
def simplest_fraction_in_interval(x, y):
    """Return the fraction with the lowest denominator in [x,y]."""
    if x == y:
        # The algorithm will not terminate if x and y are equal.
        raise ValueError("Equal arguments.")
    elif x < 0 and y < 0:
        # Handle negative arguments by solving positive case and negating.
        return -simplest_fraction_in_interval(-y, -x)
    elif x <= 0 or y <= 0:
        # One argument is 0, or arguments are on opposite sides of 0, so
        # the simplest fraction in interval is 0 exactly.
        return Fraction(0)
    else:
        # Remainder and Coefficient of continued fractions for x and y.
        xr, xc = modf(1/x);
        yr, yc = modf(1/y);
        if xc < yc:
            return Fraction(1, int(xc) + 1)
        elif yc < xc:
            return Fraction(1, int(yc) + 1)
        else:
            return 1 / (int(xc) + simplest_fraction_in_interval(xr, yr))

def approximate_fraction(x, e):
    """Return the fraction with the lowest denominator that differs
    from x by no more than e."""
    return simplest_fraction_in_interval(x - e, x + e)

def tight_layout(fig, rect=[0, 0.03, 1, 0.95]):
    """
    Reminder for myself how tight_layout works with the ect option
    fig.tight_layout(rect=rect)
    Parameters
    ----------
    fig
    rect

    Returns
    -------
    """
    fig.tight_layout(rect=rect)

## backend
def get_current_backend():
    gui = mpl.get_backend()
    print(gui)
    return gui

def list_available_backends():
    current_backend = mpl.get_backend()

    gui_backends = [i for i in mpl.rcsetup.interactive_bk]
    non_gui_backends = mpl.rcsetup.non_interactive_bk
    # gui_env = ['TKAgg', 'GTKAgg', 'Qt4Agg', 'WXAgg']

    backends = gui_backends + non_gui_backends

    available_backends = []

    print ("Non Gui backends are:", non_gui_backends)
    print ("Gui backends I will test for", gui_backends)
    for backend in backends:
        try:
            mpl.use(backend, warn=False, force=True)
            available_backends.append(backend)
        except:
            continue
    print('Available backends:')
    print(available_backends)

    mpl.use(current_backend)
    print("Currently using:", mpl.get_backend() )

def use_backend(name='agg'):
    mpl.use(name)


# smooth a curve using convolution
def smooth1d(x, window_len=11, window='hanning', log=False):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with a given signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smoothen(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.filter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smoothen() only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    if log:
        x = np.log(x)

    s = np.r_[x[window_len - 1:0:-1], x, x[-2:-window_len - 1:-1]]
    # print(len(s))
    if window == 'flat':  # moving average
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')

    y = np.convolve(w / w.sum(), s, mode='valid')
    if not log:
        return y[(window_len//2-1):(window_len//2-1)+len(x)]
    else:
        return np.exp(y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)])

def add_secondary_xaxis(ax, functions=None, loc='top', label='', log=False, **kwargs):
    """
    Adds a secondary x-axis at the top
    ... Must pass a pair of mapping functions between a current x and a new x

    e.g.
        def deg2rad(x):
            return x * np.pi / 180
        def rad2deg(x):
            return x * 180 / np.pi
        add_secondary_xaxis(ax, functions=(deg2rad, rad2deg))

    Parameters
    ----------
    ax
    functions

    Returns
    -------
    secax

    """
    if functions is None:
        print('add_secondary_xaxis: supply a mapping function (Current X to New X) and its inverse function')
        print('... e.g. (deg2rad, rad2deg)')

        def f1(x):
            return 2 * x

        def f2(x):
            return x / 2

        functions = (f1, f2)
    secax = ax.secondary_xaxis(location=loc, functions=functions)
    secax.set_xlabel(label, **kwargs)
    if log:
        secax.set_xscale("log")
    return secax

def add_secondary_yaxis(ax, functions=None, loc='right', label='', log=False, **kwargs):
    """
    Adds a secondary yaxis at the top
    ... Must pass a pair of mapping functions between a current x and a new x

    e.g.
        def deg2rad(y):
            return y * np.pi / 180
        def rad2deg(y):
            return y * 180 / np.pi
        add_secondary_yaxis(ax, functions=(deg2rad, rad2deg))

    Parameters
    ----------
    ax
    functions

    Returns
    -------
    secax

    """
    if functions is None:
        print('add_secondary_xaxis: supply a mapping function (Current X to New X) and its inverse function')
        print('... e.g. (deg2rad, rad2deg)')

        def f1(x):
            return 2 * x

        def f2(x):
            return x / 2

        functions = (f1, f2)
    secax = ax.secondary_yaxis(location=loc, functions=functions)
    secax.set_ylabel(label, **kwargs)
    if log:
        secax.set_yscale("log")
    return secax


def use_symmetric_ylim(ax):
    bottom, top = ax.get_ylim()
    if bottom * top < 0:
        bottom, top = -np.max([-bottom, top]), np.max([-bottom, top])
        ax.set_ylim(bottom=bottom, top=top)
    return ax


def make_ax_symmetric(ax, axis='y'):
    """Makes the plot symmetric about x- or y-axis"""
    """
    Makes the plot symmetric about the x- or y-axis
    
    Parameters
    ----------
    ax: axes.Axes instance
    axis: str, Choose from 'x', 'y', 'both'
    
    Returns
    -------
    None

    """
    if axis in ['y', 'both']:
        ymin, ymax = ax.get_ylim()
        yabs = max(-ymin, ymax)
        ax.set_ylim(-yabs, yabs)
    if axis in ['x', 'both']:
        xmin, xmax = ax.get_xlim()
        xabs = max(-xmin, xmax)
        ax.set_xlim(-xabs, xabs)


def make_ticks_scientific(ax, axis='both', **kwargs):
    """
    Make tick labels display in a scientific format

    Some other useful lines about tick formats
        ax.set_xticks(np.arange(0, 1.1e-3, 0.5e-3))
        ax.set_yticks(np.arange(0, 1.1e-3, 0.25e-3))
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        ax.xaxis.offsetText.set_fontsize(20)
        ax.yaxis.offsetText.set_fontsize(20)

    Parameters
    ----------
    ax: axes.Axes instance

    Returns
    -------

    """
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis=axis, **kwargs)

def match_lims(ax1, ax2, axis='both'):
    xmin1, xmax1 = ax1.get_xlim()
    ymin1, ymax1 = ax1.get_ylim()
    xmin2, xmax2 = ax2.get_xlim()
    ymin2, ymax2 = ax2.get_ylim()
    xmin, xmax = min(xmin1, xmin2), max(xmax1, xmax2)
    ymin, ymax = min(ymin1, ymin2), max(ymax1, ymax2)
    for ax in [ax1, ax2]:
        if axis in ['x', 'both']:
            ax.set_xlim(xmin, xmax)
        if axis in ['y', 'both']:
            ax.set_ylim(ymin, ymax)
def color_axis(ax, locs=['bottom', 'left', 'right'], colors=['k', 'C0', 'C1'],
               xlabel_color=None, ylabel_color=None,
               xtick_color=None, ytick_color=None):
    """
    Colors the axes (axis, ticks, and a label)

    Parameters
    ----------
    ax: axes.Axes instance
    locs: list of strings, locations of the axes. choose from 'bottom', 'left', 'right', 'top'
    colors: list of strings, colors of the axes. e.g. ['k', 'C0', 'C1']
    xlabel_color: str, color of xlabel. If None, the same colors as "colors" are used.
    ylabel_color: str, color of ylabel. If None, the same colors as "colors" are used.
    xtick_color: str, color of xtick. If None, the same colors as "colors" are used.
    ytick_color: str, color of ytick. If None, the same colors as "colors" are used.

    Returns
    -------

    """
    for loc, color in zip(locs, colors):
        ax.spines[loc].set_color(color)
        if loc in ['top', 'bottom'] and xlabel_color is None:
            xlabel_color = color
            if xlabel_color is None: xlabel_color = 'k'
        elif loc in ['right', 'left'] and ylabel_color is None:
            ylabel_color = color

    if xlabel_color is None: xlabel_color = 'k'
    if ylabel_color is None: ylabel_color = 'k'

    # match tick colors with the label colors
    if xtick_color is None: xtick_color = xlabel_color
    if ytick_color is None: ytick_color = ylabel_color

    ax.xaxis.label.set_color(xlabel_color)
    ax.tick_params(axis='x', colors=xtick_color)
    ax.xaxis.label.set_color(xlabel_color)
    ax.tick_params(axis='y', colors=ytick_color)

def pc2float(s):
    """
    Converts a percentage expression (str) to float
    e.g. pc2float(5.2%) returns 0.0052
    Parameters
    ----------
    s: str, e.g. "5.2%"

    Returns
    -------
    a floating number  (e.g. 0.0052)
    """
    return float(s.strip('%'))/100.

def float2pc(x):
    """
    Converts a float into a percentage expression
    Parameters
    ----------
    x

    Returns
    -------
    a string in float (e.g. 0.0052)
    """
    return "{0}%".format(x * 100.)



def simple_legend(ax, facecolor='white', **kwargs):
    "Removes the errorbars from the legend"
    from matplotlib import container
    handles, labels = ax.get_legend_handles_labels()
    handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    lg = ax.legend(handles, labels, **kwargs)
    frame = lg.get_frame()
    frame.set_color(facecolor)

# adjuster/reminders
## colorbar stuff
def adjust_colorbar(cb,
                    fontsize=__fontsize__,
                    label=None, labelpad=1,
                    tick_fontsize=__fontsize__,
                    ticks=None, ):
    """A helper to modify basic features of a matplotlib Colorbar object """
    if label is None:
        label = cb.ax.get_ylabel()
    cb.set_label(label, fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(ticks)
    cb.ax.tick_params(labelsize=tick_fontsize)
    cb.ax.xaxis.get_offset_text().set_fontsize(tick_fontsize)
    cb.ax.yaxis.get_offset_text().set_fontsize(tick_fontsize)


def plot_colorbar(values, cmap='viridis', colors=None, ncolors=100,
                  fignum=1, figsize=None, fig=None, ax=None, cax_spec=None,
                  orientation='vertical', label=None, labelpad=5,
                  fontsize=__fontsize__, option='normal', fformat=None,
                  ticks=None, tick_params=None, **kwargs):
    """
    Plots a stand-alone colorbar

    Parameters
    ----------
    values: 1d array-like, these values are used to create a colormap
    cmap: str, cmap object
    colors: list of colors, if given, it overwrites 'cmap'.
        ... For custom colors/colormaps, the functions below could be handy.
        ...... colors = choose_colors() # sns.choose_cubehelix_palette()
        ...... colors = get_color_list_gradient(color1='red', color2='blue') # linearly segmented colors
    ncolors
    fignum
    figsize
    orientation
    label
    labelpad
    fontsize
    option: str, if 'scientific', it uses a scientific format for the ticks
    fformat
    ticks
    tick_params
    kwargs

    Returns
    -------
    fig, cax, cb: Figure instance, axes.Axes instance, Colorbar instance
    """
    global sfmt
    if figsize is None:
        if orientation == 'horizontal':
            figsize = (7.54 * 0.5, 2)
        else:
            figsize = (2, 7.54 * 0.5)
    if cax_spec is None:
        if orientation == 'horizontal':
            cax_spec = [0.1, 0.8, 0.8, 0.1]
        else:
            cax_spec = [0.1, 0.1, 0.1, 0.8]

    if colors is not None:
        cmap = mpl.colors.LinearSegmentedColormap.from_list('cutom_cmap', colors, N=ncolors)

    fig = plt.figure(fignum, figsize=figsize)
    img = plt.imshow(np.array([values]), cmap=cmap)
    plt.gca().set_visible(False)

    cax = fig.add_axes(cax_spec)

    if option == 'scientific':
        if fformat is not None:
            sfmt.fformat = fformat
        fmt = sfmt
    else:
        if fformat is not None:
            fmt = fformat
        else:
            fmt = None
    cb = plt.colorbar(orientation=orientation, cax=cax, format=fmt, **kwargs)
    cb.set_label(label=label,
                 fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(ticks)

    if tick_params is None:
        tick_params = {'labelsize': fontsize}
    cb.ax.tick_params(**tick_params)
    # fig.tight_layout()
    return fig, cax, cb

def set_fontsize_scientific_text(ax, fontsize):
    """
    Set fontsize for the scientific format

    Parameters
    ----------
    fontsize: int

    Returns
    -------

    """
    ax.yaxis.get_offset_text().set_fontsize(fontsize)

def legend_custom(ax, colors, labels, loc=None, fontsize=__fontsize__,
                  edgecolor='k', frameon=False, **kwargs):
    """
    A custom legend with colored patches

    Parameters
    ----------
    ax
    colors
    labels
    loc
    fontsize
    edgecolor
    kwargs

    Returns
    -------

    """
    # Grab the existing legend from the axes
    # lg = ax.legend()
    handles = [mpl.patches.Patch(facecolor=color, edgecolor=edgecolor) for color in colors]
    # ax.add_artist(lg)
    new_lg = ax.legend(handles, labels, fontsize=fontsize, frameon=frameon, loc=loc, **kwargs)
    return new_lg


# Color stuff
def hex2rgb(hex, normalize=True):
    """
    Converts a HEX code to RGB in a numpy array
    Parameters
    ----------
    hex: str, hex code. e.g. #B4FBB8

    Returns
    -------
    rgb: numpy array. RGB

    """
    h = hex.strip('#')
    rgb = np.asarray(list(int(h[i:i + 2], 16) for i in (0, 2, 4)))

    if normalize:
        rgb = rgb / 255.

    return rgb

def rgb2hex(rgb):
    """
    Converts a RGB (0-1) to HEX code
    Parameters
    ----------
    rgb: tuple, list, or numpy array

    Returns
    -------
    hex: str, hex code
    """
    if isinstance(rgb[0], float):
        rgb = (int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255))
    hex = '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])
    return hex

def cname2hex(cname):
    """
    Converts a color registered on matplotlib to a HEX code
    Parameters
    ----------
    cname: str, color name

    Returns
    -------
    hex: str, hex code
    """
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS)  # dictionary. key: names, values: hex codes
    try:
        hex = colors[cname]
        return hex
    except NameError:
        print(cname, ' is not registered as default colors by matplotlib!')
        return None

def cname2rgba(cname, normalize=True):
    """
    Converts a color name to an RGBA

    Parameters
    ----------
    cname: str, color name
    normalize: bool, if True, it normalizes the RGB values to [0, 1]

    Returns
    -------
    rgba: numpy array, RGBA
    """
    hex = cname2hex(cname)
    rgba = hex2rgb(hex)
    if normalize:
        rgba = rgba / 255
    return rgba


