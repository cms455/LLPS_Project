"""
Author: Takumi Matsuzawa
Last updated: 2023/11/25
Description: This module contains useful functions for plotting.
"""

import os
import copy
from pathlib import Path
import itertools
# Graphical
import pylab as pl
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.pylab as pylab
import matplotlib.ticker as ticker
import mpl_toolkits.axes_grid1 as axes_grid
from matplotlib.offsetbox import AnchoredOffsetbox
from matplotlib.transforms import Bbox
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import mpl_toolkits
from cycler import cycler
import seaborn as sns
# Numerical
import numpy as np
from numpy import ma
from math import modf
from scipy import interpolate
from scipy.stats import binned_statistic
from scipy.optimize import curve_fit
from fractions import Fraction
# Data
import pickle
# Additional modules
import softliv.softliv.functions as func
# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Global variables
# Default color cycle: iterator which gets repeated if all elements were exhausted
#__color_cycle__ = itertools.cycle(iter(plt.rcParams['axes.prop_cycle'].by_key()['color']))
__def_colors__ = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
__color_cycle__ = itertools.cycle(__def_colors__)  #matplotlib v2.0<
__old_color_cycle__ = itertools.cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])  #matplotlib classic
__linestyle_cycle__ = itertools.cycle(['-', '--', '-.', ':'])
__softliv_colors__ = [
                        "#1A97BA",  # SoftLiv Blue (Teal)
                        "#F4A460",  # Sandy Beach (Orange)
                        "#50C878",  # Ocean Green (Green)
                        "#CC474B",  # English Vermillion (Red)
                        "#C9A0DC",  # Wisteria (Purple)
                        "#9DB3B8",  # Gull Gray
                        "#8C604D",  # Cocoa Brown
                        "#FFDB58",  # Mustard
                        "#FFC0CB",  # Soft Pink
                        "#AEE0EB",  # Light Blue
                     ]

# See all available arguments in matplotlibrc
__fontsize__ = 14
__figsize__ = (7.54*0.667, 7.54*0.667)
cmap = 'magma'
params = {'figure.figsize': __figsize__,
          'font.size': __fontsize__,  #text
        'legend.fontsize': 11
          , # legend
         'axes.labelsize': __fontsize__, # axes
         'axes.titlesize': __fontsize__,
          'lines.linewidth': 2,
         'lines.markersize': 5,

         'xtick.labelsize': __fontsize__, # tick
         'ytick.labelsize': __fontsize__,
         'xtick.major.size': 3.5, #3.5 def
         'xtick.minor.size': 2, # 2.
         'ytick.major.size': 3.5,
         'ytick.minor.size': 2,
          'figure.dpi': 500,
         }
plt.rcParams.update(params)

# FUNCTIONS
# Save a figure
def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=150, overwrite=True, tight_layout=False,
         savedata=True, transparent=True, bkgcolor='w', **kwargs):
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
    filename = "%s.%s" % (os.path.split(path)[1], ext)

    if directory == '':
        directory = '.'

    # Make sure the directory exists
    directory = Path(directory)
    if not os.path.exists(directory):
        os.makedirs(directory)

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

    # Save a fig instance. This could fail for python2.
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
            custom_cycler=False, custom_cycler_dict=None, # advanced features to change a plotting style
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

def add_subplot_custom(x0, y0 ,x1, y1):
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
    ax = pl.axes([x0, y0, x1, y1])
    return ax

# Plotting
def plot(x, y=None, fmt='-', fignum=1, figsize=None, label='', color=None, subplot=None, legend=False,
         fig=None, ax=None, maskon=False, thd=1, xmin=None, xmax=None,
         set_bottom_zero=False, symmetric=False, #y-axis
         set_left_zero=False,
         smooth=False, smoothlog=False, window_len=5, window='hanning',
         custom_cycler=None, custom_cycler_dict=None,
         return_xy=False, **kwargs):
    """
    Plot a 1d array

    Parameters
    ----------
    x: array-like, x data
    y: array-like, y data
    fmt: str, format string, e.g. 'o-', 'o--', 'o:', 'o-.', 'o', '-', '--', ':', '-.', etc.
    fignum: int, figure number
    figsize: tuple, figure size
    label: str, label for legend
    color: str, color of the plot
    subplot: int, matplotlib subplot notation. default: 111
    legend: bool, If True, it shows a legend
    fig: matplotlib.Figure instance
    ax; matplotlib.axes.Axes instance
    maskon: bool, If True, it masks erroneous data points
    thd: float, threshold for masking erroneous data points
    xmin: float, minimum x value to plot
    xmax: float, maximum x value to plot
    set_bottom_zero: bool, If True, it sets the bottom of y-axis to zero
    symmetric: bool, If True, it sets the y-axis symmetric
    set_left_zero: bool, If True, it sets the left of x-axis to zero
    smooth: bool, If True, it smooths the data by convolving with a `window`
    smoothlog: bool, If True, it smooths the data in log space
    window_len: int, smoothing window length
    window: str, smoothing window type, e.g. 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    custom_cycler: bool, If True, it enables users to customize a plot style (color cycle, marker cycle, linewidth cycle etc.)
    custom_cycler_dict: dict, A summary of a plotting style.
    return_xy: bool, If True, it returns x2plot, y2plot
    kwargs: dict, additional keyword arguments passed to plt.plot()

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    """
    if fig is None and ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    elif fig is None:
        fig = plt.gcf()
    elif ax is None:
        ax = plt.gca()
    if custom_cycler:
        apply_custom_cyclers(ax, **custom_cycler_dict)

    if y is None:
        y = copy.deepcopy(x)
        x = np.arange(len(x))
    # Make sure x and y are np.array
    x, y = np.asarray(x), np.asarray(y)

    if len(x) > len(y):
        print("Warning : x and y data do not have the same length")
        x = x[:len(y)]
    elif len(y) > len(x):
        print("Warning : x and y data do not have the same length")
        y = y[:len(x)]

    # remove nans
    keep = ~np.isnan(x) * ~np.isnan(y)
    x, y = x[keep], y[keep]

    if maskon:
        keep = identifyInvalidPoints(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x <= xmax
    if xmin is not None:
        keep *= x >= xmin
    try:
        if smooth:
            x2plot = x[keep]
            y2plot = smooth1d(y[keep], window_len=window_len, window=window)
        elif smoothlog:
            x2plot = x[keep]
            try:
                logy2plot = smooth1d(np.log10(y[keep]), window_len=window_len, window=window)
                y2plot = 10**logy2plot
            except:
                y2plot = y[keep]
        else:
            x2plot, y2plot = x[keep], y[keep]
    except:
        x2plot, y2plot = x[keep], y[keep]
    if color is None:
        ax.plot(x2plot, y2plot, fmt, label=label, **kwargs)
    else:
        ax.plot(x2plot, y2plot, fmt, color=color, label=label, **kwargs)

    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_ylim(-yabs, yabs)
    if return_xy:
        return fig, ax, x2plot, y2plot
    else:
        return fig, ax
def pdf(data, nbins=100, return_data=False, vmax=None, vmin=None, log=False,
        fignum=1, figsize=None, subplot=None, density=True, analyze=False, **kwargs):
    """
    Plots a probability distribution function of ND data
    ... a wrapper for np.histogram and matplotlib
    ... Returns fig, ax, (optional: bins, hist)

    Parameters
    ----------
    data: nd-array, list, or tuple, data used to get a histogram/pdf
    nbins: int, number of bins
    return_data: bool, If True, it returns  fig, ax, bins (centers of the bins), hist (counts or probability density values)
    vmax: float, data[data>vmax] will be ignored during counting.
    vmin: float, data[data<vmin] will be ignored during counting.
    fignum: int, figure number (the argument called "num" in matplotlib)
    figsize: tuple, figure size in inch (width x height)
    subplot: int, matplotlib subplot notation. default: 111
    density: bool, If True, it plots the probability density instead of counts.
    analyze: bool If True, it adds mean, mode, variane to the plot.
    kwargs: other kwargs passed to plot() of the velocity module

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    (Optional)
    bins: 1d array, bin centers
    hist: 1d array, probability density vales or counts
    """
    def compute_pdf(data, nbins=10, density=density):
        # Get a normalized histogram
        # exclude nans from statistics
        hist, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=density)
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, hist

    data = np.asarray(data)

    # Use data where values are between vmin and vmax
    if vmax is not None:
        cond1 = np.asarray(data) < vmax # if nan exists in data, the condition always gives False for that data point
    else:
        cond1 = np.ones(data.shape, dtype=bool)
        vmax = np.nanmax(data)
    if vmin is not None:
        cond2 = np.asarray(data) > vmin
    else:
        vmin = np.nanmin(data)
        cond2 = np.ones(data.shape, dtype=bool)
    data = data[cond1 * cond2]
    delta = (vmax - vmin) / nbins
    if log:
        bins = np.geomspace(max(vmin, 1e-16), vmax, nbins)
    else:
        bins = np.arange(vmin, vmax+delta, delta)
    # print(bins)
    # compute a pdf
    bins, hist = compute_pdf(data, nbins=bins)
    fig, ax = plot(bins, hist, fignum=fignum, figsize=figsize, subplot=subplot, **kwargs)

    if not return_data:
        return fig, ax
    else:
        return fig, ax, bins, hist


def cdf(data, nbins=100, return_data=False, vmax=None, vmin=None,
        fignum=1, figsize=None, subplot=None, **kwargs):
    """
    Plots a cummulative distribution function of ND data
    ... a wrapper for np.histogram and matplotlib
    ... Returns fig, ax, (optional: bins, hist)

    Parameters
    ----------
    data: nd-array, list, or tuple, data used to get a histogram/pdf
    nbins: int, umber of bins
    return_data: bool, If True, it returns  fig, ax, bins (centers of the bins), hist (counts or probability density values)
    vmax: float, data[data>vmax] will be ignored during counting.
    vmin: float, data[data<vmin] will be ignored during counting.
    fignum: int, figure number (the argument called "num" in matplotlib)
    figsize: tuple, figure size in inch (width x height)
    subplot: int, matplotlib subplot notation. default: 111
    density: bool, If True, it plots the probability density instead of counts.
    analyze: bool If True, it adds mean, mode, variane to the plot.
    kwargs: other kwargs passed to plot() of the velocity module

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    (Optional)
    bins: 1d array, bin centers
    hist: 1d array, probability density vales or counts
    """
    def compute_pdf(data, nbins=10):
        """
        Compute probability distribution function of data

        Parameters
        ----------
        data: nd-array, list, or tuple, data used to get a histogram/pdf
        nbins: int, number of bins

        Returns
        -------
        bins: 1d array, bin centers
        pdf: 1d array, probability density vales or counts
        """
        # Get a normalized histogram
        # exclude nans from statistics
        pdf, bins = np.histogram(data.flatten()[~np.isnan(data.flatten())], bins=nbins, density=True)
        # Get middle points for plotting sake.
        bins1 = np.roll(bins, 1)
        bins = (bins1 + bins) / 2.
        bins = np.delete(bins, 0)
        return bins, pdf

    def compute_cdf(data, nbins=10):
        """
        Compute cummulative distribution function of data

        Parameters
        ----------
        data: nd-array, list, or tuple, data used to get a histogram/pdf
        nbins: int, number of bins

        Returns
        -------
        bins: 1d array, bin centers
        cdf: 1d array, cummulative distribution function
        """
        bins, pdf = compute_pdf(data, nbins=nbins)
        cdf = np.cumsum(pdf) * np.diff(bins, prepend=0)
        return bins, cdf

    data = np.asarray(data)

    # Use data where values are between vmin and vmax
    if vmax is not None:
        cond1 = np.asarray(data) < vmax # if nan exists in data, the condition always gives False for that data point
    else:
        cond1 = np.ones(data.shape, dtype=bool)
    if vmin is not None:
        cond2 = np.asarray(data) > vmin
    else:
        cond2 = np.ones(data.shape, dtype=bool)
    data = data[cond1 * cond2]

    # compute a cdf
    bins, cdf = compute_cdf(data, nbins=nbins)
    fig, ax = plot(bins, cdf, fignum=fignum, figsize=figsize, subplot=subplot, **kwargs)

    if not return_data:
        return fig, ax
    else:
        return fig, ax, bins, cdf


def errorbar(x, y, xerr=None, yerr=0., fignum=1, marker='o', fillstyle='full', linestyle='None', label=None, mfc='white',
             subplot=None, legend=False, legend_remove_bars=False, figsize=None, maskon=False, thd=1, capsize=10,
             xmax=None, xmin=None, ax=None, **kwargs):
    """
    Plot a 1d array with error bars

    Parameters
    ----------
    x: array-like, x data
    y: array-like, y data
    xerr: array-like, x error
    yerr: array-like, y error
    fignum: int, figure number
    marker: str, marker style
    fillstyle: str, fill style
    linestyle: str, line style
    label: str, label for legend
    mfc: str, marker face color
    subplot: int, matplotlib subplot notation. default: 111
    legend: bool, If True, it shows a legend
    legend_remove_bars: bool, If True, it removes bars from the legend
    figsize: tuple, figure size
    maskon: bool, If True, it masks erroneous data points
    thd: float, threshold for masking erroneous data points, default: 1
    capsize: float, length of the error bar caps in points, default: 10
    xmax: float, maximum x value to plot
    xmin: float, minimum x value to plot
    ax: matplotlib.axes.Axes instance, If None, it creates a new figure and axes
    kwargs: dict, additional keyword arguments passed to plt.errorbar()

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x, y = np.array(x), np.array(y)

    # MASK FEATURE
    if maskon:
        keep = identifyInvalidPoints(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin

    # Make xerr and yerr numpy arrays if they are not scalar. Without this, TypeError would be raised.
    if xerr is not None:
        if not ((isinstance(xerr, int) or isinstance(xerr, float))):
            xerr_ = np.array(xerr)
        else:
            xerr_ = np.ones_like(x) * xerr
        xerr_[xerr_==0] = np.nan
        xerr_ = xerr_[keep]
    else:
        xerr_ = xerr
    if yerr is not None:
        if not ((isinstance(yerr, int) or isinstance(yerr, float))):
            yerr_ = np.array(yerr)
        else:
            yerr_ = np.ones_like(x) * yerr
        yerr_[yerr_==0] = np.nan
        yerr_ = yerr_[keep]
    else:
        yerr_ = yerr

    if fillstyle == 'none':
        ax.errorbar(x[keep], y[keep], xerr=xerr_, yerr=yerr_, marker=marker, mfc=mfc, linestyle=linestyle,
                    label=label, capsize=capsize, **kwargs)
    else:
        ax.errorbar(x[keep], y[keep], xerr=xerr, yerr=yerr, marker=marker, fillstyle=fillstyle,
                    linestyle=linestyle, label=label, capsize=capsize,  **kwargs)

    if legend:
        ax.legend()

        if legend_remove_bars:
            from matplotlib import container
            handles, labels = ax.get_legend_handles_labels()
            handles = [h[0] if isinstance(h, container.ErrorbarContainer) else h for h in handles]
    return fig, ax

def errorfill(x, y, yerr, fignum=1, color=None, subplot=None, alpha_fill=0.3, ax=None, label=None,
              legend=False, figsize=None, maskon=False, thd=1,
              xmin=None, xmax=None, smooth=False, smoothlog=False, window_len=5, window='hanning',
              set_bottom_zero=False, set_left_zero=False, symmetric=False, return_xy=False,
              **kwargs):
    """
    Plot a 1d array with a filled region between the error bars

    Parameters
    ----------
    x: array-like, x data
    y: array-like, y data
    yerr: array-like, y error
    fignum: int, figure number
    color: str, color of the plot
    subplot: int, matplotlib subplot notation. default: 111
    alpha_fill: float, alpha value of the filled region, default: 0.3
    ax: matplotlib.axes.Axes instance, If None, it creates a new figure and axes
    label: str, label for legend
    legend: bool, If True, it shows a legend
    figsize: tuple, figure size
    maskon: bool, If True, it masks erroneous data points
    thd: float, threshold for masking erroneous data points, default: 1
    xmin: float, minimum x value to plot
    xmax: float, maximum x value to plot
    smooth: bool, If True, it smooths the data by convolving with a `window`
    smoothlog: bool, If True, it smooths the data in the log space
    window_len: int, smoothing window length
    window: str, smoothing window type, e.g. 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
    set_bottom_zero: bool, If True, it sets the bottom of y-axis to zero
    set_left_zero: bool, If True, it sets the left of x-axis to zero
    symmetric: bool, If True, it sets the y-axis symmetric
    return_xy: bool, If True, it returns the plotted values of x and y
    kwargs: dict, additional keyword arguments passed to plt.plot()

    Returns
    -------
    fig: matplotlib.Figure instance
    ax: matplotlib.axes.Axes instance
    x2plot: 1d array, x values used for plotting, after masking and smoothing, optional
    y2plot: 1d array, y values used for plotting, after masking and smoothing, optional
    """

    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    x = np.array(x)
    y = np.array(y)
    yerr = np.array(yerr)

    #ax = ax if ax is not None else plt.gca()
    # if color is None:
    #     color = color_cycle.next()
    if maskon:
        keep = identifyInvalidPoints(x, y, thd=thd)
    else:
        keep = [True] * len(x)
    if xmax is not None:
        keep *= x < xmax
    if xmin is not None:
        keep *= x >= xmin

    mask2removeNans = ~np.isnan(x) * ~np.isnan(y)
    keep = keep * mask2removeNans

    if smooth:
        x2plot = x[keep]
        y2plot = smooth1d(y[keep], window_len=window_len, window=window)
    elif smoothlog:
        x2plot = x[keep]
        try:
            logy2plot = smooth1d(np.log10(y[keep]), window_len=window_len, window=window)
            y2plot = 10**logy2plot
        except:
            y2plot = y[keep]
    else:
        x2plot, y2plot = x[keep], y[keep]
    if len(yerr) == len(y):
        ymin = y2plot - yerr[keep]
        ymax = y2plot + yerr[keep]
    elif len(yerr) == 2:
        yerrdown, yerrup = yerr
        ymin = y2plot - yerrdown
        ymax = y2plot + yerrup
    else:
        ymin = y2plot - yerr
        ymax = y2plot + yerr


    p = ax.plot(x2plot, y2plot, color=color, label=label, **kwargs)
    color = p[0].get_color()
    ax.fill_between(x2plot, ymax, ymin, color=color, alpha=alpha_fill)

    #patch used for legend
    color_patch = mpatches.Patch(color=color, label=label)
    if legend:
        plt.legend(handles=[color_patch])

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_ylim(-yabs, yabs)

    if not return_xy:
        return fig, ax, color_patch
    else:
        return fig, ax, color_patch, x2plot, y2plot


def bin_and_errorbar(x_, y_, xerr=None,
                     n_bins=100, mode='linear', bin_center=True, return_std=False,
                     fignum=1, ax=None, marker='o', fillstyle='full',
                     linestyle='None', linewidth=1, label=None, mfc='white',
                     subplot=None, legend=False, figsize=None, maskon=False, thd=1, capsize=5,
                     set_bottom_zero=False, symmetric=False, #y-axis
                     set_left_zero=False,
                     return_stats=False, **kwargs):
    """
    Takes scattered data points (x, y), bin them (compute avg and std), then plots the results with error bars

    Parameters
    ----------
    x : array-like
    y : array-like
    xerr: must be a scalar or numpy array with shape (N,1) or (2, N)... [xerr_left, xerr_right]
        ... if xerr==0, it removes the error bars in x.
    yerr:  must be a scalar or numpy array with shape (N,) or (2, N)... [yerr_left, yerr_right]
    n_bins: int, number of bins used to compute a histogram between xmin and xmax
    mode: str, default: 'linear', options are 'linear' and 'log'. Select either linear binning or logarithmic binning
        ... If "linear", it computes statistics using evenly separated bins between xmin and xmax.
        ... If "log", it uses bins evenly separted in the log space. (It assumes that xmin>0)
            i.e. The bin edges are like (10^-1.0, 10^-0.5),  (10^-0.5, 10^0),  (10^0, 10^0.5), and so on.
    bin_center: bool, default: True.
        ... passed to get_binned_stats()
    return_std: bool, default: False.
        ... passed to get_binned_stats()
        ... If False, it uses standard errors as error bars, instead of using standard deviations
    fignum: int, figure number
    ax: Axes object, default: None
        ... If given, this becomes the Axes on which the results are plotted
    marker: str, default: 'o', marker style
    fillstyle: str, default: 'full'. Options: 'full', 'none'. See matplotlib scatter for more details
    linestyle: str, default:'None'
    linewidth: float, linewidth of the error bars
    label: str, label for a legend
    mfc: str, default:'white', marker face color
        ... Use this with fillstyle='none' in order to change the face color of the marker.
        ... Common usage: empty circles- fillstyle='none', mfc='white'
    subplot: int, three-digit number. e.g.-111
    legend: bool, default: False. If True, ax.legend is called at the end.
    figsize: tuple, figure size in inches
    maskon: bool, default: False
        ... This hides "suspicious" data points / outliers.
        ... See the docstr of identifyInvalidPoints() for more details
    thd: float, threshold value used for identifyInvalidPoints() to determine the outliers
    capsize: float, width of the error bars
    return_stats: bool, default: False
        ... If True, it returns the binned results (that are being plotted): x[mask], y[mask], xerr[mask], yerr[mask]
    kwargs: passed to ax.errorbar()

    Returns
    -------
    If not return_stats (default),
        fig, ax: a Figure instance, an Axes instance
    If return_stats:
        fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]: a Figure instance, an Axes instance, binned results (x, y, x_err, y_err)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x_, y_ = np.array(x_), np.array(y_)
    x, y, yerr = get_binned_stats(x_, y_, n_bins=n_bins, mode = mode, bin_center=bin_center, return_std = return_std)
    if xerr is None:
        xerr = np.ones_like(x) * (x[1] - x[0]) / 2.
    elif type(xerr) in [int, float]:
        xerr = np.ones_like(x) * xerr
    xerr[xerr == 0] = np.nan
    yerr[yerr == 0] = np.nan

    if maskon:
        mask = identifyInvalidPoints(x, y, thd=thd)
    else:
        mask = [True] * len(x)
    if fillstyle == 'none':
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, mfc=mfc, linestyle=linestyle,
                    label=label, capsize=capsize, linewidth=linewidth, **kwargs)
    else:
        ax.errorbar(x[mask], y[mask], xerr=xerr[mask], yerr=yerr[mask], marker=marker, fillstyle=fillstyle,
                    linestyle=linestyle, label=label, capsize=capsize, linewidth=linewidth,   **kwargs)
    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        xmin, xmax = ax.get_xlim()
        xabs = np.abs(max(-xmin, xmax))
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-yabs, yabs)

    if not return_stats: # default
        return fig, ax
    else:
        return fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]

def bin_and_scatter(x_, y_, xerr=None,
                     n_bins=100, mode='linear', bin_center=True, return_std=False,
                     fignum=1, ax=None, marker='o', fillstyle='full',
                     linestyle='None', linewidth=1, label=None, mfc='white',
                     subplot=None, legend=False, figsize=None, maskon=False, thd=1, capsize=5,
                     set_bottom_zero=False, symmetric=False, #y-axis
                     set_left_zero=False,
                     return_stats=False, **kwargs):
    """
    Takes scattered data points (x, y), bin them (compute avg and std), then plots the results with error bars

    Parameters
    ----------
    x : array-like
    y : array-like
    xerr: must be a scalar or numpy array with shape (N,1) or (2, N)... [xerr_left, xerr_right]
        ... if xerr==0, it removes the error bars in x.
    yerr:  must be a scalar or numpy array with shape (N,) or (2, N)... [yerr_left, yerr_right]
    n_bins: int, number of bins used to compute a histogram between xmin and xmax
    mode: str, default: 'linear', options are 'linear' and 'log'. Select either linear binning or logarithmic binning
        ... If "linear", it computes statistics using evenly separated bins between xmin and xmax.
        ... If "log", it uses bins evenly separted in the log space. (It assumes that xmin>0)
            i.e. The bin edges are like (10^-1.0, 10^-0.5),  (10^-0.5, 10^0),  (10^0, 10^0.5), and so on.
    bin_center: bool, default: True.
        ... passed to get_binned_stats()
    return_std: bool, default: False.
        ... passed to get_binned_stats()
        ... If False, it uses standard errors as error bars, instead of using standard deviations
    fignum: int, figure number
    ax: Axes object, default: None
        ... If given, this becomes the Axes on which the results are plotted
    marker: str, default: 'o', marker style
    fillstyle: str, default: 'full'. Options: 'full', 'none'. See matplotlib scatter for more details
    linestyle: str, default:'None'
    linewidth: float, linewidth of the error bars
    label: str, label for a legend
    mfc: str, default:'white', marker face color
        ... Use this with fillstyle='none' in order to change the face color of the marker.
        ... Common usage: empty circles- fillstyle='none', mfc='white'
    subplot: int, three-digit number. e.g.-111
    legend: bool, default: False. If True, ax.legend is called at the end.
    figsize: tuple, figure size in inches
    maskon: bool, default: False
        ... This hides "suspicious" data points / outliers.
        ... See the docstr of identifyInvalidPoints() for more details
    thd: float, threshold value used for identifyInvalidPoints() to determine the outliers
    capsize: float, width of the error bars
    return_stats: bool, default: False
        ... If True, it returns the binned results (that are being plotted): x[mask], y[mask], xerr[mask], yerr[mask]
    kwargs: passed to ax.errorbar()

    Returns
    -------
    If not return_stats (default),
        fig, ax: a Figure instance, an Axes instance
    If return_stats:
        fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]: a Figure instance, an Axes instance, binned results (x, y, x_err, y_err)
    """
    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()

    # Make sure that xerr and yerr are numpy arrays
    ## x, y, xerr, yerr do not have to be numpy arrays. It is just a convention. - takumi 04/01/2018
    x_, y_ = np.array(x_), np.array(y_)
    x, y, yerr = get_binned_stats(x_, y_, n_bins=n_bins, mode = mode, bin_center=bin_center, return_std = return_std)
    if xerr is None:
        xerr = np.ones_like(x) * (x[1] - x[0]) / 2.
    elif type(xerr) in [int, float]:
        xerr = np.ones_like(x) * xerr
    xerr[xerr == 0] = np.nan
    yerr[yerr == 0] = np.nan

    if maskon:
        mask = identifyInvalidPoints(x, y, thd=thd)
    else:
        mask = [True] * len(x)
    ax.scatter(x[mask], y[mask], marker=marker,
                    linestyle=linestyle, label=label, **kwargs)
    if legend:
        ax.legend()

    if set_bottom_zero:
        ax.set_ylim(bottom=0)
    if set_left_zero:
        ax.set_xlim(left=0)
    if symmetric:
        xmin, xmax = ax.get_xlim()
        xabs = np.abs(max(-xmin, xmax))
        ymin, ymax = ax.get_ylim()
        yabs = np.abs(max(-ymin, ymax))
        ax.set_xlim(-xabs, xabs)
        ax.set_ylim(-yabs, yabs)

    if not return_stats: # default
        return fig, ax
    else:
        return fig, ax, x[mask], y[mask], xerr[mask], yerr[mask]


# Colorbar
class FormatScalarFormatter(mpl.ticker.ScalarFormatter):
    """
    Ad-hoc class to subclass matplotlib.ticker.ScalarFormatter
    in order to alter the number of visible digits on color bars
    """

    def __init__(self, fformat="%03.1f", offset=True, mathText=True):
        self.fformat = fformat
        mpl.ticker.ScalarFormatter.__init__(self, useOffset=offset, useMathText=mathText)
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
    global sfmt
    sfmt = FormatScalarFormatter()  # Default format: "%04.1f"
    # sfmt.fformat = fformat # update format
    # sfmt._set_format() # this updates format for scientific nota
    sfmt._update_format(fformat)

# Update the formatting into a scientific paper style
reset_sfmt()
def get_sfmt(fformat="%03.1f"):
    ""
    global sfmt
    reset_sfmt(fformat=fformat)
    return sfmt


## 2D plotsFor the plot you showed at group meeting of lambda converging with resolution, can you please make a version with two x axes (one at the top, one below) one pixel spacing, other PIV pixel spacing, and add a special tick on each for the highest resolution point.
# (pcolormesh)
def color_plot(x, y, z,
               subplot=None, fignum=1, figsize=None, ax=None,
               vmin=None, vmax=None, log10=False, label=None,
               cbar=True, cmap='magma', symmetric=False, enforceSymmetric=True,
               aspect='equal', option='scientific', ntick=5, tickinc=None,
               crop=None, fontsize=None, ticklabelsize=None, cb_kwargs={}, return_cb=False,
               **kwargs):
    """

    Parameters
    ----------
    x: 2d array
    y: 2d array
    z: 2d array
    subplot: int, default is 111
    fignum
    figsize
    ax
    vmin
    vmax
    log10
    label
    cbar
    cmap
    symmetric
    aspect: str, 'equal' or 'auto
    option
    ntick
    tickinc
    crop
    kwargs
    cb_kwargs: dict, kwargs for add_colorbar()
        ... e.g. {"fformat": %.0f}
    return_cb: bool, default: False
        ... if True, this function returns fig, ax, cc, cb (colorbar instance)

    Returns
    -------
    fig:
    ax:
    cc: QuadMesh instance
    cb: colorbar instance (optional)
    """

    if ax is None:
        fig, ax = set_fig(fignum, subplot, figsize=figsize)
    else:
        fig = plt.gcf()
        # fig, ax = set_fig(fignum, subplot, figsize=figsize, aspect=aspect)
    if crop is not None:
        x = x[crop:-crop, crop:-crop]
        y = y[crop:-crop, crop:-crop]
        z = z[crop:-crop, crop:-crop]


    if log10:
        z = np.log10(z)

    # For Diverging colormap, ALWAYS make the color thresholds symmetric
    symCmap1 = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
    symCmap2 = [c + '_r' for c in symCmap1]
    symCmap = symCmap1 + symCmap2
    if cmap in symCmap \
            and enforceSymmetric:
        symmetric = True

    if symmetric:
        hide = np.isinf(z)
        keep = ~hide
        if vmin is None and vmax is None:
            v = max(np.abs(np.nanmin(z[keep])), np.abs(np.nanmax(z[keep])))
            vmin, vmax = -v, v
        elif vmin is not None and vmax is not None:
            arr = np.asarray([vmin, vmax])
            v = np.nanmax(np.abs(arr))
            vmin, vmax = -v, v
        elif vmin is not None and vmax is None:
            vmax = -vmin
        else:
            vmin = -vmax




    # Note that the cc returned is a matplotlib.collections.QuadMesh
    # print('np.shape(z) = ' + str(np.shape(z)))
    if vmin is None and vmax is None:
        # plt.pcolormesh returns a QuadMesh class object.
        cc = ax.pcolormesh(x, y, z, cmap=cmap, shading='nearest', **kwargs)
    else:
        cc = ax.pcolormesh(x, y, z, cmap=cmap, vmin=vmin, vmax=vmax, shading='nearest', **kwargs)

    if cbar:
        if vmin is None and vmax is None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        elif vmin is not None and vmax is None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        elif vmin is None and vmax is not None:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmax=vmax, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)
        else:
            cb = add_colorbar(cc, ax=ax, label=label, option=option, vmin=vmin, vmax=vmax, ntick=ntick, tickinc=tickinc, fontsize=fontsize, ticklabelsize=ticklabelsize, **cb_kwargs)

    ax.set_aspect(aspect)
    # set edge color to face color
    cc.set_edgecolor('face')

    if return_cb and cbar:
        return fig, ax, cc, cb
    else:
        return fig, ax, cc

def create_cax(ax, location='right', size='5%', pad=0.15, **kwargs):
    divider = axes_grid.make_axes_locatable(ax)
    cax = divider.append_axes(location, size=size, pad=pad, **kwargs)
    return cax

def add_colorbar(mappable, fig=None, ax=None, cax=None, fignum=None, location='right', label=None, fontsize=None,
                 option='normal',
                 tight_layout=True, ticklabelsize=None, aspect='equal', ntick=5, tickinc=None,
                 size='5%', pad=0.15, caxAspect=None, fformat="%03.1f", labelpad=1, **kwargs):
    """
    Adds a color bar

    e.g.
        fig = plt.figure()
        img = fig.add_subplot(111)
        ax = img.imshow(im_data)
        colorbar(ax)
    Parameters
    ----------
    mappable
    location

    Returns
    -------

    """
    global sfmt

    def get_ticks_for_sfmt(mappable, n=10, inc=0.5, **kwargs):
        """
        Returns ticks for scientific notation
        ... setting format=smft sometimes fails to use scientific notation for colorbar.
        ... This function should ensure the colorbar object to have appropriate ticks
         to display numbers in scientific fmt once the generated ticks are passed to fig.colorbar().
        Parameters
        ----------
        mappable
        n: int, (ROUGHLY) the number of ticks
        inc: float (0, 0.5]
        ... 0.5 or 0.25 is recommended
        Returns
        -------
        ticks, list, ticks for scientific format
        """
        # ticks for scientific notation
        zmin, zmax = np.nanmin(mappable.get_array()), np.nanmax(mappable.get_array())
        if 'vmin' in kwargs.keys():
            zmin = kwargs['vmin']
        if 'vmax' in kwargs.keys():
            zmax = kwargs['vmax']

        exponent = int(np.floor(np.log10(np.abs(zmax))))
        if tickinc is not None:
            # Specify the increment of ticks!
            dz = inc * 10 ** exponent
            ticks = [i * dz for i in range(int(zmin / dz), int(zmax / dz) + 1)]
        else:
            # Specify the number of ticks!
            exp = int(np.floor(np.log10((zmax - zmin) / n)))
            dz = np.round((zmax - zmin) / n, -exp)
            ticks = [i * dz for i in range(int(zmin / dz), int(zmax / dz) + 1)]

        return ticks

    def remove_vmin_vmax_from_kwargs(**kwargs):
        if 'vmin' in kwargs.keys():
            del kwargs['vmin']
        if 'vmax' in kwargs.keys():
            del kwargs['vmax']
        return kwargs

    # fig = ax.figure
    # Get a Figure instance
    if fig is None:
        fig = plt.gcf()
        if fignum is not None:
            fig = plt.figure(num=fignum)
    if cax is None:
        # ax = mappable.axes
        if ax is None:
            ax = plt.gca()
        divider = axes_grid.make_axes_locatable(ax)
        cax = divider.append_axes(location, size=size, pad=pad, )

    reset_sfmt(fformat=fformat)

    if caxAspect is not None:
        cax.set_aspect(caxAspect)
    if option == 'scientific_custom':
        ticks = get_ticks_for_sfmt(mappable, n=ntick, inc=tickinc, **kwargs)
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        # sfmt.format = '$\mathdefault{%1.1f}$'
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, ticks=ticks, **kwargs)
        # cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    elif option == 'scientific':
        # old but more robust
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        cb = fig.colorbar(mappable, cax=cax, format=sfmt, **kwargs)
    else:
        kwargs = remove_vmin_vmax_from_kwargs(**kwargs)
        cb = fig.colorbar(mappable, cax=cax, **kwargs)

    if not label is None:
        if fontsize is None:
            cb.set_label(label, labelpad=labelpad)
        else:
            cb.set_label(label, labelpad=labelpad, fontsize=fontsize)
    if ticklabelsize is not None:
        cb.ax.tick_params(labelsize=ticklabelsize)
        cb.ax.yaxis.get_offset_text().set_fontsize(ticklabelsize)
    # ALTERNATIVELY
    # global __fontsize__
    # cb.ax.tick_params(axis='both', which='major', labelsize=__fontsize__, length=5, width=0.2)
    # cb.ax.yaxis.get_offset_text().set_fontsize(__fontsize__) # For scientific format

    # Adding a color bar may distort the aspect ratio. Fix it.
    if ax is not None:
        if aspect == 'equal':
            ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb


def add_discrete_colorbar(ax, colors, vmin=0, vmax=None, label=None, fontsize=None, option='normal',
                          tight_layout=True, ticklabelsize=None, ticklabel=None,
                          aspect=None, useMiddle4Ticks=False, **kwargs):
    fig = ax.get_figure()
    if vmax is None:
        vmax = len(colors)
    tick_spacing = (vmax - vmin) / float(len(colors))
    if not useMiddle4Ticks:
        vmin, vmax = vmin - tick_spacing / 2., vmax - tick_spacing / 2.
    ticks = np.linspace(vmin, vmax, len(colors) + 1) + tick_spacing / 2.  # tick positions

    # if there are too many ticks, just use 3 ticks
    if len(ticks) > 10:
        n = len(ticks)
        ticks = [ticks[0], ticks[n // 2] - 1, ticks[-2]]
        if ticklabel is not None:
            ticklabel = [ticklabel[0], ticklabel[n / 2], ticklabel[-1]]

    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable

    if option == 'scientific':
        cb = fig.colorbar(sm, ticks=ticks, format=sfmt, **kwargs)
    else:
        cb = fig.colorbar(sm, ticks=ticks, **kwargs)

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
    if aspect == 'equal':
        ax.set_aspect('equal')

    # Adding a color bar may disport the overall balance of the figure. Fix it.
    if tight_layout:
        fig.tight_layout()

    return cb


def add_colorbar_alone(ax, values, cax=None, cmap=cmap, label=None, fontsize=None, option='normal', fformat=None,
                       tight_layout=True, ticklabelsize=None, ticklabel=None,
                       aspect=None, location='right', color='k',
                       size='5%', pad=0.15, **kwargs):
    """
    Add a colorbar to a figure without a mappable
    ... It creates a dummy mappable with given values

    ... LOCATION OF CAX
    fig, ax = graph.set_fig(1, 111)
    w, pad, size = 0.1, 0.05, 0.05
    graph.add_colorbar_alone(ax, [0, 1], pad=float2pc(pad), size=float2pc(size), tight_layout=False)
    graph.add_subplot_axes(ax, [1-w-(1-1/(1.+pad+size)), 0.8, w, 0.2])


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
        cb = fig.colorbar(sm, cax=cax, **kwargs)

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


def colorbar(fignum=None, label=None, fontsize=__fontsize__):
    """
    Use is DEPRICATED. This method is replaced by add_colorbar(mappable)
    I keep this method for old codes which might have used this method
    Parameters
    ----------
    fignum :
    label :

    Returns
    -------
    """
    fig, ax = set_fig(fignum)
    c = plt.colorbar()
    if not label == None:
        c.set_label(label, fontsize=fontsize)
    return c


def create_colorbar(values, cmap='viridis', figsize=None, orientation='vertical', label='qty (mm)', fontsize=11,
                    labelpad=0, ticks=None, **kwargs):
    """
    Creates a horizontal/vertical colorbar for reference using pylab.colorbar()

    Parameters
    ----------
    values: 1d array-like, used to specify the min and max of the colorbar
    cmap: cmap instance or str, default: 'viridis'
    figsize: tuple, figure size in inches, default: None
    orientation: str, 'horizontal' or 'vertical'
    label: str, label of the color bar
    fontsize: fontsize for the label and the ticklabel
    labelpad: float, padding for the label
    ticks: 1d array, tick locations

    Returns
    -------

    """
    values = np.array(values)
    if len(values.shape) == 1:
        values = np.expand_dims(values, axis=0)
    #     values = np.array([[-1.5, 1]])
    if orientation == 'horizontal' and figsize is None:
        figsize = (7.54 * 0.5, 1)
    elif orientation == 'vertical' and figsize is None:
        figsize = (1, 7.54 * 0.5)
    fig = pl.figure(figsize=figsize)
    img = pl.imshow(values, cmap=cmap)
    ax = pl.gca()
    ax.set_visible(False)

    if orientation == 'horizontal':
        cax = pl.axes([0.1, 0.8, 0.8, 0.1])
    else:
        cax = pl.axes([0.8, 0.1, 0.1, 0.8])
    cb = pl.colorbar(orientation=orientation, cax=cax, **kwargs)
    cb.set_label(label=label, fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(np.arange(-1.5, 1.5, 0.5))
    cb.ax.tick_params(labelsize=fontsize)
    fig.tight_layout()
    return fig, ax, cb


def dummy_scalarMappable(values, cmap):
    """
    Returns a dummy scalarMappable that can be used to make a stand-alone color bar
    e.g.
        sm = dummy_scalarMappable([0, 100], 'viridis')
        fig = plt.figure(1)
        fig.colorbar(sm, pad=0.1)
    Parameters
    ----------
    values: list, array, this is used to specify the range of the color bar
    cmap: str, cmap object

    Returns
    -------
    sm
    """
    vmin, vmax = np.nanmin(values), np.nanmax(values)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])  # dummy mappable
    return sm


### Axes
# Label
def labelaxes(ax, xlabel, ylabel, **kwargs):
    ax.set_xlabel(xlabel, **kwargs)
    ax.set_ylabel(ylabel, **kwargs)


# multi-color labels
def labelaxes_multicolor(ax, list_of_strings, list_of_colors, axis='x', anchorpad=0, **kwargs):
    """this function creates axes labels with multiple colors
    ax specifies the axes object where the labels should be drawn
    list_of_strings is a list of all of the text items
    list_if_colors is a corresponding list of colors for the strings
    axis='x', 'y', or 'both' and specifies which label(s) should be drawn"""
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker, VPacker

    # x-axis label
    if axis == 'x' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', **kwargs))
                 for text, color in zip(list_of_strings, list_of_colors)]
        xbox = HPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_xbox = AnchoredOffsetbox(loc=3, child=xbox, pad=anchorpad, frameon=False, bbox_to_anchor=(0.2, -0.09),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_xbox)

    # y-axis label
    if axis == 'y' or axis == 'both':
        boxes = [TextArea(text, textprops=dict(color=color, ha='left', va='bottom', rotation=90, **kwargs))
                 for text, color in zip(list_of_strings[::-1], list_of_colors)]
        ybox = VPacker(children=boxes, align="center", pad=0, sep=5)
        anchored_ybox = AnchoredOffsetbox(loc=3, child=ybox, pad=anchorpad, frameon=False, bbox_to_anchor=(-0.2, 0.4),
                                          bbox_transform=ax.transAxes, borderpad=0.)
        ax.add_artist(anchored_ybox)


# Limits
def setaxes(ax, xmin, xmax, ymin, ymax, **kwargs):
    ax.set_xlim(xmin, xmax, **kwargs)
    ax.set_ylim(ymin, ymax, **kwargs)
    return ax


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
    if xmin == 0:
        ax.set_xlim(xmin, xmax)
    if ymin == 0:
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
             rect=[0, 0.03, 1, 1.05],
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
def set_standard_pos(ax):
    """
    Sets standard positions for added texts in the plot
    left: 0.025, right: 0.75
    bottom: 0.10 top: 0.90
    xcenter: 0.5 ycenter:0.5
    Parameters
    ----------
    ax

    Returns
    -------
    top, bottom, right, left, xcenter, ycenter: float, position

    """
    left_margin, right_margin, bottom_margin, top_margin = 0.025, 0.75, 0.1, 0.90

    xleft, xright = ax.get_xlim()
    ybottom, ytop = ax.get_ylim()
    width, height = np.abs(xright - xleft), np.abs(ytop - ybottom)

    if ax.get_xscale() == 'linear':
        left, right = xleft + left_margin * width, xleft + right_margin * width
        xcenter = xleft + width / 2.
    if ax.get_yscale() == 'linear':
        bottom, top = ybottom + bottom_margin * height, ybottom + top_margin * height
        ycenter = ybottom + height / 2.

    if ax.get_xscale() == 'log':
        left, right = xleft + np.log10(left_margin * width), xleft + np.log10(right_margin * width)
        xcenter = xleft + np.log10(width / 2.)

    if ax.get_yscale() == 'log':
        bottom, top = ybottom + np.log10(bottom_margin * height), ybottom + np.log10(top_margin * height)
        ycenter = ybottom + np.log10(height / 2.)

    return top, bottom, right, left, xcenter, ycenter, height, width


def addtext(ax, text, transform=None, fontsize=__fontsize__, color=None, **kwargs):
    """
    Adds text to a plot. You can specify the position where the texts will appear by 'option'

    Parameters
    ----------
    ax
    text
    transform
    fontsize
    color
    kwargs

    Returns
    ax : with a text
    -------

    """
    if transform is None:
        transform = ax.transAxes
    ax.text(x, y, text, transform=transform, fontsize=fontsize, color=color, **kwargs)
    return ax


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
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder,
                               linewidth=linewidth)
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


##Clear plot
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


def countcolorcycle(color_cycle=__color_cycle__):
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
                         marker=['o', 'o', 'o', 'o'], s=[0, 0, 0, 0], **kwargs):
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
    custom_cycler = cycler(color=color) + cycler(linestyle=linestyle) + cycler(lw=linewidth) + cycler(
        marker=marker) + cycler(markersize=s)
    ax.set_prop_cycle(custom_cycler)


def create_cmap_using_values(colors=None, color1='greenyellow', color2='darkgreen', color3=None, n=100):
    """
    Create a colormap instance from a list
    ... same as mpl.colors.LinearSegmentedColormap.from_list()
    Parameters
    ----------
    colors
    color1
    color2
    n

    Returns
    -------

    """
    if colors is None:
        colors = get_color_list_gradient(color1=color1, color2=color2, color3=color3, n=n)
    cmap_name = 'new_cmap'
    newcmap = mpl.colors.LinearSegmentedColormap.from_list(cmap_name, colors, N=n)
    return newcmap


def create_cmap_from_colors(colors_list, name='newmap'):
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
        color1_rgb = hex2rgb(color1) / 255.  # np array
        color2_rgb = hex2rgb(color2) / 255.  # np array

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
        color1_rgb = hex2rgb(color1) / 255.  # np array
        color2_rgb = hex2rgb(color2) / 255.  # np array
        color3_rgb = hex2rgb(color3) / 255.  # np array

        n_middle = int((n - 1) / 2)

        r1 = np.linspace(color1_rgb[0], color2_rgb[0], n_middle, endpoint=False)
        g1 = np.linspace(color1_rgb[1], color2_rgb[1], n_middle, endpoint=False)
        b1 = np.linspace(color1_rgb[2], color2_rgb[2], n_middle, endpoint=False)
        color_list1 = list(zip(r1, g1, b1))

        r2 = np.linspace(color2_rgb[0], color3_rgb[0], n - n_middle)
        g2 = np.linspace(color2_rgb[1], color3_rgb[1], n - n_middle)
        b2 = np.linspace(color2_rgb[2], color3_rgb[2], n - n_middle)
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
        colors = graph.choose_colors()
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
    colors = dict(mpl.colors.BASE_COLORS, **mpl.colors.CSS4_COLORS)  # dictionary. key: names, values: hex codes
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
    Update a default matplotlib setting
    ... plt.rcParams.update(params)

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
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


# plotting styles
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


#
def get_markers():
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
    x_labelsize *= rect[2] ** 0.3
    y_labelsize *= rect[3] ** 0.3
    subax.xaxis.set_tick_params(labelsize=x_labelsize)
    subax.yaxis.set_tick_params(labelsize=y_labelsize)
    if spines2hide is not None:
        subax.spines[spines2hide].set_visible(False)
    return subax


def create_inset_ax(ax, rect, axisbg='w', alpha=1, **kwargs):
    """DEPRICATED. Use add_subplot_axes"""
    return add_subplot_axes(ax, rect, axisbg=axisbg, alpha=alpha, **kwargs)


# sketches
def draw_circle(ax, x, y, r, linewidth=1, edgecolor='r', facecolor='none', fill=False, **kwargs):
    """
    Draws a circle on the axes (ax)

    Parameters
    ----------
    ax: matplotlib axes object
    x: float
    y: float
    r: float
    linewidth: float
    edgecolor:
    facecolor
    fill
    kwargs

    Returns
    -------

    """
    circle = plt.Circle((x, y), r, linewidth=linewidth, edgecolor=edgecolor, facecolor=facecolor, fill=fill, **kwargs)
    ax.add_artist(circle)
    return circle


def draw_rectangle(ax, x, y, width, height, angle=0.0, linewidth=1, edgecolor='r', facecolor='none', **kwargs):
    """
    Draws a rectangle in a figure (ax)
    Parameters
    ----------
    ax
    x
    y
    width
    height
    angle
    linewidth
    edgecolor
    facecolor
    kwargs

    Returns
    -------

    """
    rect = mpatches.Rectangle((x, y), width, height, angle=angle, linewidth=linewidth, edgecolor=edgecolor,
                              facecolor=facecolor, **kwargs)
    ax.add_patch(rect)
    ax.axis('equal')  # this ensures to show the rectangle if the rectangle is bigger than the original size
    return rect


def draw_box(ax, xx, yy, w_box=351., h_box=351., xoffset=0, yoffset=0, linewidth=5,
             scalebar=True, sb_length=50., sb_units=' mm', sb_loc=(0.95, 0.1), sb_txtloc=(0.0, 0.4),
             sb_lw=10, sb_txtcolor='white', fontsize=None,
             facecolor='k', fluidcolor=None,
             bounding_box=True, bb_lw=1, bb_color='w'):
    """
    Draws a box and fills the surrounding area with color (default: skyblue)
    Adds a scalebar by default
    ... drawn box center coincides with the center of given grids(xx, yy)
    ... in order to shift the center of the box, use xoffset any yoffset
    Parameters
    ----------
    ax: matplotlib.axes.Axes instance
    xx: 2d numpy array
        x coordinates
    yy: 2d numpy array
        y coordinates
    w_box: float/int
        width of the box- used to be set as 325
    h_box: float/int
        height of the box- used to be set as 325
    xoffset: float/int
        real number to shift the box center in the x direction
    yoffset:
        real number to shift the box center in the x direction
    linewidth: int
        linewidth of drawn box
    scalebar: bool (default: True)
        ... draws a scalebar inside the drawn box
    sb_length: int
        ... length of the scale bar in physical units.
        ...... In principle, this can be float. If you want that, edit the code where ax.text() is called.
        ...... Generalizing to accept the float requires a format which could vary everytime, so just accept integer.
    sb_units: str
        ... units of the sb_length. Default: '$mm$'
    sb_loc: tuple, (x, y)
        ... location of the scale bar. Range: [0, 1]
        ... the units are with respect the width and height of the box
    sb_txtloc: tuple, (x, y)
        ... location of the TEXT of the scale bar. Range: [0, 1]
        ... x=0: LEFT of the scale bar, x=1: RIGHT of the scale bar
        ... y=0: LEFT of the scale bar, x=1: RIGHT of the scale bar

    sb_lw: float
        ... line width of the scale bar

    facecolor
    fluidcolor

    Returns
    -------

    """
    xmin, xmax = np.nanmin(xx), np.nanmax(xx)
    ymin, ymax = np.nanmin(yy), np.nanmax(yy)
    # if np.nanmean(yy) > 0:
    #     xc, yc = xmin + (xmax - xmin) / 2., ymin + (ymax - ymin) / 2.
    # else:
    #     xc, yc = xmin + (xmax - xmin) / 2., ymin - (ymax - ymin) / 2.
    xc, yc = xmin + (xmax - xmin) / 2., ymin + (ymax - ymin) / 2.
    x0, y0 = xc - w_box / 2. + xoffset, yc - h_box / 2. + yoffset
    draw_rectangle(ax, x0, y0, w_box, h_box, linewidth=linewidth, facecolor=facecolor, zorder=0)
    if fluidcolor is not None:
        ax.set_facecolor(fluidcolor)

    if bounding_box:
        w, h = xmax - xmin, ymax - ymin
        dx, dy = np.abs(xx[0, 1] - xx[0, 0]), np.abs(yy[1, 0] - yy[0, 0])
        draw_rectangle(ax, xmin - dx, ymin - dy, width=w + 2 * dx, height=h + 2 * dy, edgecolor=bb_color,
                       linewidth=bb_lw)

    if scalebar:
        dx, dy = np.abs(xx[0, 1] - xx[0, 0]), np.abs(yy[1, 0] - yy[0, 0])  # mm/px

        #         x0_sb, y0_sb = x0 + 0.8 * w_box, y0 + 0.1*h_box
        x1_sb, y1_sb = x0 + sb_loc[0] * w_box, y0 + sb_loc[1] * h_box
        x0_sb, y0_sb = x1_sb - sb_length, y1_sb
        if sb_loc[1] < 0.5:
            x_sb_txt, y_sb_txt = x0_sb + sb_txtloc[0] * sb_length, y0 + sb_loc[1] * h_box * sb_txtloc[1]
        else:
            x_sb_txt, y_sb_txt = x0_sb + sb_txtloc[0] * sb_length, y0 - (1 - sb_loc[1]) * h_box * sb_txtloc[1] + sb_loc[
                1] * h_box
        x_sb, y_sb = [x0_sb, x1_sb], [y0_sb, y1_sb]
        xmin, xmax, ymin, ymax = ax.axis()
        width, height = xmax - xmin, ymax - ymin
        ax.plot(x_sb, y_sb, linewidth=sb_lw, color=sb_txtcolor)
        if fontsize is None or fontsize > 0:
            ax.text(x_sb_txt, y_sb_txt, '%d%s' % (sb_length, sb_units), color=sb_txtcolor, fontsize=fontsize)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)


def draw_cuboid(ax, xx, yy, zz, color='c', lw=2, **kwargs):
    """Draws a cuboid (projection='3d')"""

    xmin, xmax = np.nanmin(xx), np.nanmax(xx)
    ymin, ymax = np.nanmin(yy), np.nanmax(yy)
    zmin, zmax = np.nanmin(zz), np.nanmax(zz)
    rx = [xmin, xmax]
    ry = [ymin, ymax]
    rz = [zmin, zmax]
    w, h, d = xmax - xmin, ymax - ymin, zmax - zmin
    for s, e in itertools.combinations(np.array(list(itertools.product(rx, ry, rz))), 2):
        dist = np.linalg.norm(s - e)
        if dist in [w, h, d]:
            ax.plot3D(*zip(s, e), color=color, lw=lw, **kwargs)

    ax.set_xlim(rx)
    ax.set_ylim(ry)
    ax.set_zlim(rz)
    set_axes_equal(ax)


def draw_sphere(ax, xc, yc, zc, r, color='r', **kwargs):
    # Make data
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = xc + r * np.outer(np.cos(u), np.sin(v))
    y = yc + r * np.outer(np.sin(u), np.sin(v))
    z = zc + r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Plot the surface
    surf = ax.plot_surface(x, y, z, color=color, **kwargs)
    set_axes_equal(ax)


def draw_sphere_wireframe(ax, xc, yc, zc, r, color='r', lw=1, **kwargs):
    """
    Draws a sphere using a wireframe
    Parameters
    ----------
    ax
    xc
    yc
    zc
    r
    color
    lw
    kwargs

    Returns
    -------

    """
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = r * np.cos(u) * np.sin(v) + xc
    y = r * np.sin(u) * np.sin(v) + yc
    z = r * np.cos(v) + zc
    ax.plot_wireframe(x, y, z, color=color, lw=lw, **kwargs)
    set_axes_equal(ax)


def add_color_wheel(fig=None, fignum=1, figsize=__figsize__,
                    rect=[0.68, 0.65, 0.2, 0.2],
                    cmap=None, cmapname='hsv',
                    norm=None, values=[-np.pi, np.pi],
                    n=2056,
                    ring=True,
                    text='Phase',
                    fontsize=__fontsize__,
                    ratio=1, text_loc_ratio=0.35, text_loc_angle=np.pi * 1.07,
                    **kwargs
                    ):
    if fig is None:
        fig = plt.figure(num=fignum, figsize=figsize)

    subax = fig.add_axes(rect, projection='polar')
    subax._direction = 2 * np.pi

    if cmap is None or norm is None:
        colors, cmap, norm = get_colors_and_cmap_using_values(values, cmap=cmapname, n=n)

    cb = mpl.colorbar.ColorbarBase(subax,
                                   cmap=cmap,
                                   norm=norm,
                                   orientation='horizontal')

    # aesthetics - get rid of border and axis labels
    cb.outline.set_visible(False)
    subax.set_axis_off()

    if ring:
        w = values[1] - values[0]
        subax.set_rlim([values[0] - w * ratio, values[1]])  # This makes it a color RING not a wheel (filled circle)
    # addtext(subax, text, np.pi*1.07, (values[0] - w/2.9 ), color='w', fontsize=fontsize)
    addtext(subax, text,
            text_loc_angle,
            values[0] - w * ratio * (1 / (1 / text_loc_ratio * ratio + 1)), color='w', fontsize=fontsize)

    # subax2 = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    # plot([0, 1], [0, 1], ax=subax2)
    # addtext(subax2, text, 0, 0, color='w', fontsize=fontsize)
    # # print(values[0], values[0] - w)
    return subax, cb


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


def identifyInvalidPoints(x, y, thd=1):
    """
    Retruns a mask that can be sued to hide erroneous data points for 1D plots
    ... e.g. x[mask] and y[mask] hide the jumps which appear to be false to human eyes
    ... Uses P = dy/dx / y to determine whether data points appear to be false
        If P is high, we'd expect a jump. thd is a threshold of P.

    Parameters
    ----------
    x: 1d array
    y: 1d array
    thd: float, threshold on P (fractional dy/dx)

    Returns
    -------
    mask: 1d bool array

    """
    # remove nans
    keep_x, keep_y = ~np.isnan(x), ~np.isnan(y)
    keep = keep_x * keep_y
    x, y = x[keep], y[keep]

    fractional_dydx = np.gradient(y, x) / y
    reasonable_rate_of_change = np.abs(
        fractional_dydx) < thd  # len(reasonable_rate_of_change) is not necessarily equal to len(keep)
    reasonable_rate_of_change = np.roll(reasonable_rate_of_change,
                                        1)  # shift the resulting array (the convention of np.gradient)
    keep[keep] = reasonable_rate_of_change
    return keep


def tight_layout(fig, rect=[0, 0.03, 1, 0.95]):
    """
    Reminder for myself how tight_layout works with the ect option
    fig.tight_layout(rect=rect)
    Parameters
    ----------
    fig: matplotlib.Figure object
    rect: list, [left, bottom, right, top]

    Returns
    -------
    None
    """
    fig.tight_layout(rect=rect)


## Interactive plotting
class LineDrawer(object):
    """
    Class which allows users to draw lines/splines by clicking pts on the plot
        ... Default: lines/splines are closed.
        ... make sure that matplotlib backend is interactive

    Procedure for self.draw_lines() or self.draw_splines:
        It uses plt.ginput()
        1. Add a point by a left click
        2. Remove a point by a right click
        3. Stop interaction (move onto the next line to draw)

    Example
        # Pass matplotlib.axes._subplots.AxesSubplot object whose coordinates are used for extracting pts
        ld = LineDrawer(ax)

        # Draw lines/splines
        ld.draw_lines(n=5) # Draw 5 lines (connecting 5 set of points)
        # ld.draw_splines(n=2) # Or draw 2 splines based on the clicked points

        xs, ys = ld.xs, ld.ys # Retrieve x and y coords of pts used to draw lines/splines
        # xis, yis = ld.xis, ld.yis # Retrieve x and y coords for each spline

        # plot the first contour
        plt.plot(xs[0], ys[0]

        # for example, I could feed this contour to compute a line integral using vel.compute_circulation()


    """

    def __init__(self, ax):
        self.ax = ax

    def get_contour(self, npt=100, close=True):
        ax = self.ax
        xy = plt.ginput(npt)

        x = [p[0] for p in xy]
        y = [p[1] for p in xy]
        #         line = ax.scatter(x,y, marker='x', s=20, zorder=100)
        #         ax.figure.canvas.draw()
        #         self.lines.append(line)

        if close:
            # append the starting x,y coordinates
            x = np.r_[x, x[0]]
            y = np.r_[y, y[0]]

        self.x = x
        self.y = y

        return x, y

    def draw_lines(self, n=1, close=True):
        ax = self.ax
        xs, ys = [], []
        for i in range(n):
            x, y = self.get_contour(close=close)
            xs.append(x)
            ys.append(y)

            ax.plot(x, y)

        self.xs = xs
        self.ys = ys

    def spline_fit(self, x, y, n=1000):
        from scipy import interpolate
        # fit splines to x=f(u) and y=g(u), treating both as periodic. also note that s=0
        # is needed in order to force the spline fit to pass through all the input points.
        tck, u = interpolate.splprep([x, y], s=0, per=True)

        # evaluate the spline fits for 1000 evenly spaced distance values
        xi, yi = interpolate.splev(np.linspace(0, 1, n), tck)

        return xi, yi

    def draw_splines(self, n=1, npt=100, n_sp=1000, close=True):
        ax = self.ax

        xs, ys = [], []
        xis, yis = [], []
        for i in range(n):
            x, y = self.get_contour(npt=npt, close=close)
            xi, yi = self.spline_fit(x, y, n=n_sp)

            xs.append(x)
            ys.append(y)

            xis.append(xi)
            yis.append(yi)

            # ax.plot(x, y)
            ax.plot(xi, yi)

        self.xs = xs
        self.ys = ys
        self.xis = xis
        self.yis = yis

    def return_pts_on_splines(self):
        return self.xis, self.yis

    def close(self):
        plt.close()


class PointFinder(object):
    def __init__(self, ax, xx, yy, weight=None):
        self.ax = ax
        self.xx = xx
        self.yy = yy
        self.ind = None

        if weight is None:
            self.weight = np.ones_like(xx)
        else:
            self.weight = weight

    def get_pts(self, npt=100):
        def find_indices(xx, yy, xs, ys):
            xg, yg = xx[0, :], yy[:, 0]
            xmin, xmax, ymin, ymax = xg.min(), xg.max(), yg.min(), yg.max()
            # i_list, j_list = [], []
            inds = []
            for n in range(len(xs)):
                if xs[n] > xmin and xs[n] < xmax and ys[n] > ymin and ys[n] < ymax:

                    X = np.abs(xg - xs[n])
                    Y = np.abs(yg - ys[n])
                    j = int(np.where(X == X.min())[0])
                    i = int(np.where(Y == Y.min())[0])
                    # i_list.append(i)
                    # j_list.append(j)
                else:
                    i, j = np.nan, np.nan
                inds.append(np.asarray([i, j]))
            return inds

        ax = self.ax
        xy = plt.ginput(npt)
        x = [p[0] for p in xy]
        y = [p[1] for p in xy]

        inds = find_indices(self.xx, self.yy, x, y)
        self.ind = inds
        self.x = x
        self.y = y
        return x, y, inds

    def find_local_center_of_mass(self, kernel_radius=2):
        def get_subarray(arr, i, j, kernel_radius):
            arr = np.asarray(arr)
            nrows, ncols = arr.shape

            imax = i + kernel_radius
            imin = i - kernel_radius
            jmax = j + kernel_radius
            jmin = j - kernel_radius

            if imax >= nrows:
                imax = nrows - 1
            if imin < 0:
                imin = 0
            if jmax >= ncols:
                jmax = ncols - 1
            if jmin < 0:
                jmin = 0
            subarr = arr[imin:imax, jmin:jmax]
            return subarr

        xcs, ycs = [], []

        for n, idx in enumerate(self.ind):
            if ~np.isnan(idx[0]):
                xx_sub = get_subarray(self.xx, idx[0], idx[1], kernel_radius=kernel_radius)
                yy_sub = get_subarray(self.yy, idx[0], idx[1], kernel_radius=kernel_radius)
                weight_sub = get_subarray(self.weight, idx[0], idx[1], kernel_radius=kernel_radius)

                xc = np.nansum(xx_sub * weight_sub) / np.nansum(weight_sub)
                yc = np.nansum(yy_sub * weight_sub) / np.nansum(weight_sub)
            else:
                xc, yc = np.nan, np.nan
            xcs.append(xc)
            ycs.append(yc)

            self.ax.scatter([xc], [yc], marker='x', color='k')
        self.xc = xcs
        self.yc = ycs

        return xcs, ycs

    def get_local_center_of_mass(self, npt=100, kernel_radius=2):
        x, y, inds = self.get_pts(npt=npt)
        xcs, ycs = self.find_local_center_of_mass(kernel_radius=kernel_radius)
        return xcs, ycs

    # def get_local_center_of_mass(self, weight, kernel_size=3):
    #     from scipy import ndimage
    #     import numpy as np
    #     arr_conv = ndimage.generic_filter(weight, np.nanmean, size=kernel_size,
    #                                       mode='constant', cval=np.NaN)


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

    print("Non Gui backends are:", non_gui_backends)
    print("Gui backends I will test for", gui_backends)
    for backend in backends:
        try:
            mpl.use(backend, warn=False, force=True)
            available_backends.append(backend)
        except:
            continue
    print('Available backends:')
    print(available_backends)

    mpl.use(current_backend)
    print("Currently using:", mpl.get_backend())


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
        return y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)]
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
    ax: matplotlib.axes.Axes instance
    functions: tuple, (f1, f2), default: None, e.g. (lambda x: 2*x, lambda x: x/2)
        f1: function, mapping from current x to new x
        f2: function, mapping from new x to current x
    loc: str, default: 'top', location of the secondary x-axis
    label: str, default: '', label of the secondary x-axis
    log: bool, default: False, whether to use log scale
    kwargs, default: None, keyword arguments for ax.secondary_xaxis()

    Returns
    -------
    secax: matplotlib.axes.Axes instance
        secondary x-axis
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
    Adds a secondary y-axis at the right
    ... Must pass a pair of mapping functions between a current y and a new y
    e.g.
        def deg2rad(y):
            return y * np.pi / 180
        def rad2deg(y):
            return y * 180 / np.pi
        add_secondary_yaxis(ax, functions=(deg2rad, rad2deg))

    Parameters
    ----------
    ax: matplotlib.axes.Axes instance
    functions: tuple, (f1, f2), default: None, e.g. (lambda y: 2*y, lambda y: y/2)
    loc: str, default: 'right', location of the secondary y-axis
    label: str, default: '', label of the secondary y-axis
    log: bool, default: False, whether to use log scale
    kwargs: keyword arguments for ax.secondary_yaxis()

    Returns
    -------
    secax: matplotlib.axes.Axes instance
        secondary y-axis
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


def make_symmetric_ylim(ax):
    """
    Makes the y-axis symmetric about 0

    Parameters
    ----------
    ax: matplotlib.axes.Axes instance

    Returns
    -------
    ax: matplotlib.axes.Axes instance
    """
    bottom, top = ax.get_ylim()
    if bottom * top < 0:
        bottom, top = -np.max([-bottom, top]), np.max([-bottom, top])
        ax.set_ylim(bottom=bottom, top=top)
    return ax


def get_binned_stats(arg, var, n_bins=100, mode='linear', bin_center=True, return_std=False):
    """
    Make a histogram out of a pair of 1d arrays.
    ... Returns arg_bins, var_mean, var_err
    ... The given arrays could contain nans and infs. They will be ignored.

    Parameters
    ----------
    arg: 1d array, controlling variable
    var: 1d array, data array to be binned
    n_bins: int, default: 100
    mode: str, deafult: 'linear'
        If 'linear', var will be sorted to equally spaced bins. i.e. bin centers increase linearly.
        If 'log', the bins will be not equally spaced. Instead, they will be equally spaced in log.
        ... bin centers will be like... 10**0, 10**0.5, 10**1.0, 10**1.5, ..., 10**9
    return_std: bool
        If True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N-1)
    Returns
    -------
    arg_bins: 1d array, bin centers
    var_mean: 1d array, mean values of data in each bin
    var_err: 1d array, standard error of data in each bin, (std / sqrt(N-1))
    """

    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1: 1d array, same length as arr2
        arr2: 1d array, same length as arr1

        Returns
        -------
        arr1: 1d array, sorted arr1
        arr2: 1d array, sorted arr2 in the order of arr1
        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return np.asarray(arr1), np.asarray(arr2)

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U
        Parameters
        ----------
        U: N-d array, could contain nan and inf

        Returns
        -------
        mask: N-d array, True for nan and inf values
        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    arg, var = np.asarray(arg), np.asarray(var)

    # make sure rr and corr do not contain nans
    mask1 = get_mask_for_nan_and_inf(arg)
    mask1 = ~mask1
    mask2 = get_mask_for_nan_and_inf(var)
    mask2 = ~mask2
    mask = mask1 * mask2

    if mode == 'log':
        argmin, argmax = np.nanmin(arg), np.nanmax(arg)
        mask_for_log10arg = get_mask_for_nan_and_inf(np.log10(arg))
        exp_min, exp_max = np.nanmin(np.log10(arg)[~mask_for_log10arg]), np.nanmax(np.log10(arg)[~mask_for_log10arg])
        exp_interval = (exp_max - exp_min) / n_bins
        exp_bin_centers = np.linspace(exp_min, exp_max, n_bins)
        exp_bin_edges = np.append(exp_bin_centers, exp_max + exp_interval) - exp_interval / 2.
        bin_edges = 10 ** (exp_bin_edges)
        bins = bin_edges
        mask_for_arg = get_mask_for_nan_and_inf(bins)
        bins = bins[~mask_for_arg]
    else:
        bins = n_bins

    # get a histogram
    if not bin_center:
        arg_means, arg_edges, binnumber = binned_statistic(arg[mask], arg[mask], statistic='mean', bins=bins)
    var_mean, bin_edges, binnumber = binned_statistic(arg[mask], var[mask], statistic='mean', bins=bins)
    var_err, _, _ = binned_statistic(arg[mask], var[mask], statistic='std', bins=bins)
    counts, _, _ = binned_statistic(arg[mask], var[mask], statistic='count', bins=bins)

    # bin centers
    if mode == 'log':
        bin_centers = 10 ** ((exp_bin_edges[:-1] + exp_bin_edges[1:]) / 2.)
    else:
        binwidth = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - binwidth / 2

    # Sort arrays
    if bin_center:
        arg_bins, var_mean = sort2arr(bin_centers, var_mean)
        arg_bins, var_err = sort2arr(bin_centers, var_err)
    else:
        arg_bins, var_mean = sort2arr(arg_means, var_mean)
        arg_bins, var_err = sort2arr(arg_means, var_err)
    if return_std:
        return arg_bins, var_mean, var_err
    else:
        return arg_bins, var_mean, var_err / np.sqrt(counts - 1)


def make_ax_symmetric(ax, axis='y'):
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
    None
    """
    ax.ticklabel_format(style='sci', scilimits=(0, 0), axis=axis, **kwargs)


def match_lims(ax1, ax2, axis='both'):
    """
    Matches the limits of two axes
    
    Parameters
    ----------
    ax1: axes.Axes instance
    ax2: axes.Axes instance
    axis: str, Choose from 'x', 'y', 'both'

    Returns
    -------
    None
    """
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
    Colors the axes (spines, ticks, and labels)

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


def smoothen(x, window_len=11, window='hanning', log=False):
    """smoothen the data using a window with requested size.

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
        return y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)]
    else:
        return np.exp(y[(window_len // 2 - 1):(window_len // 2 - 1) + len(x)])


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
    return float(s.strip('%')) / 100.


def float2pc(x):
    """
    Converts a float into a percentage expression
    Parameters
    ----------
    x: float, e.g. 0.0052

    Returns
    -------
    a string in float (e.g. 0.0052)
    """
    return "{0}%".format(x * 100.)


def removeErrorbarsInLegend(ax, facecolor='white', **kwargs):
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
    """
    Adjusts the colorbar

    Parameters
    ----------
    cb: colorbar instance
    fontsize: int, default: __fontsize__
    label: str, default: None
    labelpad: int, default: 1
    tick_fontsize: int, default: __fontsize__
    ticks: list, default: None

    Returns
    -------
    None
    """
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
    Plots a colorbar given a list of values

    Parameters
    ----------
    values: list of values
    cmap: str, default: 'viridis'
    colors: list of colors, default: None, If given, cmap will be ignored. Must match the number of values.
    ncolors: int, default: 100
    fignum: int, default: 1
    figsize: tuple, default: None
    fig: matplotlib.figure.Figure instance, default: None, If None, a new figure will be created
    ax: matplotlib.axes.Axes instance, default: None, If None, a new axes will be created
    cax_spec: list, default: None, Position and size of the color bar, [left, bottom, width, height]
    orientation" str, default: 'vertical', Choose from 'vertical' and 'horizontal'
    label: str, default: None, label of the colorbar
    labelpad: int, default: 5, labelpad of the colorbar
    fontsize: int, default: __fontsize__
    option: str, default: 'normal', Choose from 'normal' and 'scientific'
    fformat: str, default: None, format of the colorbar tick labels, e.g. '%.1e', default: graph.sfmt.fformat
    ticks: list, default: None, tick positions
    tick_params: dict, default: None, keyword arguments for colorbar.ax.tick_params()
    kwargs: keyword arguments for matplotlib.pyplot.colorbar()

    Returns
    -------
    fig: matplotlib.figure.Figure instance
    cax: matplotlib.axes.Axes instance
    cb: matplotlib.colorbar.Colorbar instance
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

    fig = pl.figure(fignum, figsize=figsize)
    img = pl.imshow(np.array([values]), cmap=cmap)
    pl.gca().set_visible(False)

    cax = pl.axes(cax_spec)

    if option == 'scientific':
        if fformat is not None:
            sfmt.fformat = fformat
        fmt = sfmt
    else:
        if fformat is not None:
            fmt = fformat
        else:
            fmt = None
    cb = pl.colorbar(orientation=orientation, cax=cax, format=fmt, **kwargs)
    cb.set_label(label=label,
                 fontsize=fontsize, labelpad=labelpad)
    if ticks is not None:
        cb.set_ticks(ticks)

    if tick_params is None:
        tick_params = {'labelsize': fontsize}
    cb.ax.tick_params(**tick_params)
    # fig.tight_layout()
    return fig, cax, cb


def resample(x, y, n=100, mode='linear'):
    """
    It resamples x and y.
    ... this is useful to crete a evenly spaced data in log from a linearly spaced data, and vice versa

    Parameters
    ----------
    x: 1d array
    y: 1d array
    n: int, number of points to resample
    mode: str, options are "linear" and "log"

    Returns
    -------
    x_new, y_rs: 1d arrays of new x and new y
    """

    def get_mask_for_nan_and_inf(U):
        """
        Returns a mask for nan and inf values in a multidimensional array U

        Parameters
        ----------
        U: N-d array

        Returns
        -------

        """
        U = np.array(U)
        U_masked_invalid = ma.masked_invalid(U)
        return U_masked_invalid.mask

    # x, y = copy.deepcopy(x_), copy.deepcopy(y_)
    x, y = np.array(x), np.array(y)  # np.array creates a new object unlike np.asarray

    # remove nans and infs
    hidex = get_mask_for_nan_and_inf(x)
    hidey = get_mask_for_nan_and_inf(y)
    keep = ~hidex * ~hidey
    x, y = x[keep], y[keep]

    xmin, xmax = np.nanmin(x), np.nanmax(x)
    if mode == 'log':
        if xmax < 0:
            raise ValueError('... log sampling cannot be performed as the max. value of x is less than 0')
        else:
            if xmin > 0:
                keep = [True] * len(x)
            else:
                keep = x > 0  # ignore data points s.t. x < 0
            xmin = np.nanmin(x[keep])
            logx = np.log10(x[keep])
            logxmin, logxmax = np.log10(xmin), np.log10(xmax)
            logx_new = np.linspace(logxmin, logxmax, n, endpoint=True)
            x_new = 10 ** logx_new
            flog = interpolate.interp1d(logx, y[keep])
            y_rs = flog(logx_new)
            return x_new, y_rs
    elif mode == 'loglog':
        if xmax < 0:
            raise ValueError('... log sampling cannot be performed as the max. value of x is less than 0')
        else:
            if xmin > 0:
                keep = [True] * len(x)
            else:
                keep = x > 0  # ignore data points s.t. x < 0
        xmin = np.nanmin(x[keep])
        logx = np.log10(x[keep])
        logxmin, logxmax = np.log10(xmin), np.log10(xmax)
        logx_new = np.linspace(logxmin, logxmax, n, endpoint=True)
        x_new = 10 ** logx_new
        flog = interpolate.interp1d(logx, np.log10(y[keep]))
        y_rs = 10 ** flog(logx_new)
        return x_new, y_rs
    else:
        x_new = np.linspace(xmin, xmax, n, endpoint=True)
        f = interpolate.interp1d(x, y)
        y_rs = f(x_new)

        return x_new, y_rs

def set_fontsize_scientific_text(ax, fontsize):
    """
    Sets the fontsize of the scientific text in the axes

    Parameters
    ----------
    fontsize: int, fontsize

    Returns
    -------
    None
    """
    ax.yaxis.get_offset_text().set_fontsize(fontsize)