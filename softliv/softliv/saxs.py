# Author: Takumi Matsuzawa
# Last Modified: 2023/11/22
# Description: A module to handle saxs experiment data

import os
import sys
import pathlib
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage, interpolate, signal, special
from scipy.signal import butter, medfilt, filtfilt  # filters for signal processing
from scipy.stats import binned_statistic, binned_statistic_2d, binned_statistic_dd
import copy
import re
from tqdm import tqdm
from itertools import groupby
import bioxtasraw.RAWAPI as raw # RAW API

# Fundamentals
def natural_sort(arr):
    """
    natural-sorts elements in a given array
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)

    e.g.-  arr = ['a28', 'a01', 'a100', 'a5']
    ... WITHOUT natural sorting,
     -> ['a01', 'a100', 'a28', 'a5']
    ... WITH natural sorting,
     -> ['a01', 'a5', 'a28', 'a100']


    Parameters
    ----------
    arr: list or numpy array of strings

    Returns
    -------
    sorted_array: natural-sorted
    """

    def atoi(text):
        'natural sorting'
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split('(\d+)', text)]

    return sorted(arr, key=natural_keys)

def get_binned_stats(arg, var, n_bins=100, mode='linear',
                     statistic='mean',
                     bin_center=True, return_std=False):
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
        If True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N)
    Returns
    -------
    arg_bins: 1d array, bin centers
    var_mean: 1d array, mean values of data in each bin
    var_err: 1d array, error of data in each bin
        ... If return_std=True, it returns the STD of the statistics instead of the error = STD / np.sqrt(N)

    """

    def sort2arr(arr1, arr2):
        """
        Sort arr1 and arr2 using the order of arr1
        e.g. a=[2,1,3], b=[9,1,4] -> a[1,2,3], b=[1,9,4]
        Parameters
        ----------
        arr1
        arr2

        Returns
        -------
        Sorted arr1, and arr2

        """
        arr1, arr2 = list(zip(*sorted(zip(arr1, arr2))))
        return np.asarray(arr1), np.asarray(arr2)

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
        arg_means, arg_edges, binnumber = binned_statistic(arg[mask], arg[mask], statistic=statistic, bins=bins)
    try:
        var_mean, bin_edges, binnumber = binned_statistic(arg[mask], var[mask], statistic=statistic, bins=bins)
        var_std, _, _ = binned_statistic(arg[mask], var[mask], statistic='std', bins=bins)
        counts, _, _ = binned_statistic(arg[mask], var[mask], statistic='count', bins=bins)
    except:
        arg_means = var_mean = np.asarray([np.nan] * n_bins)
        var_std = np.asarray([np.nan] * n_bins)
        arg_edges = bin_edges = np.asarray([np.nan] * (n_bins+1))

        counts = np.asarray([np.nan] * n_bins)
    # bin centers
    if mode == 'log':
        bin_centers = 10 ** ((exp_bin_edges[:-1] + exp_bin_edges[1:]) / 2.)
    else:
        binwidth = (bin_edges[1] - bin_edges[0])
        bin_centers = bin_edges[1:] - binwidth / 2

    # Sort arrays
    if bin_center:
        arg_bins, var_mean = sort2arr(bin_centers, var_mean)
        arg_bins, var_std = sort2arr(bin_centers, var_std)
    else:
        arg_bins, var_mean = sort2arr(arg_means, var_mean)
        arg_bins, var_std = sort2arr(arg_means, var_std)
    if return_std:
        return arg_bins, var_mean, var_std
    else:
        return arg_bins, var_mean, var_std / np.sqrt(counts)


# READING Bio-SAXS Data (CHESS)
def get_scattering_profile(file, settings=None, path2settings=None):
    """
    Returns a scattering profile from a given directory

    Parameters
    ----------
    dpath: str, path to the directory containing the scattering data
    settings: bioxtasraw.RAWSettings.RawGuiSettings instance, default: None
        ... If None, it loads settings from `path2settings`
    path2settings: str, default: None, path to the settings file (.cfg)

    Returns
    -------
    q: 1d array, wavenumber
    intensity: 1d array, azimuthally averaged scattering intensity
    """
    if settings is None:
        settings = raw.load_settings(path2settings)
    profile = raw.load_and_integrate_images(dpath, settings)
    q = profile[0][0].q # q = 4 pi sin(theta) / lambda
    intensity = profile[0][0].i # scattering intesntiy
    return q, intensity


def get_averaged_scattering_profile(files, settings=None, path2settings=None):
    '''
    Creates a spectrum for each file in list of files
    Computes the average and returns the spectrum
    '''
    if settings is None:
        settings = raw.load_settings(path2settings)
    profiles = raw.load_and_integrate_images(files, settings)[0]
    avg_profile = raw.average(profiles)
    q = avg_profile.q
    intensity_avg = avg_profile.i
    return q, intensity_avg

def process_SAXS_data(dataDir, outputDir=None, settings=None):
    """
    Processes all the SAXS data in the dataDir
    ... dataDir must contain .h5 files.
    ...... dataDir / SAMPLE_NAME_data_0001.h5
    ...... dataDir / SAMPLE_NAME_data_0002.h5, ...

    Parameters
    ----------
    dataDir: str, path to the directory containing the data
    outputDir: str, path to the directory where the output will be stored
    settings: bioxtasraw.RAWSettings.RawGuiSettings instance, default: None

    Returns
    -------
    df: pandas DataFrame, a table containing the processed data
    """
    def get_prefix(filename):
        return "_".join(filename.split('_')[:-1])

    if settings is None:
        settings_path = search_cfg_file(dataDir)
        settings = raw.load_settings(settings_path)

    files = natural_sort(os.listdir(dataDir))
    dataFiles = [file for file in files if file.endswith('.h5') and 'data' in file]

    # Grouping the file names
    groupedFiles = [list(group) for key, group in groupby(sorted(dataFiles, key=get_prefix), get_prefix)]

    scatteringData = {}
    sampleNames, qs, intensities = [], [], []
    for group in tqdm(groupedFiles):
        try:
            grpname = group[0].split('_data')[0]
            groupPaths = [dataDir / file for file in group]
            q, iq =  get_averaged_scattering_profile(groupPaths, settings=settings)
            sampleNames.append(grpname)
            qs.append(q)
            intensities.append(iq)
        except:
            print('Error in processing %s. Skipping...' % grpname)
            sampleNames.append(grpname)
            qs.append(None) # Missing Data
            intensities.append(None) # Missing Data

    df_saxs = pd.DataFrame({'Sample': sampleNames, 'q': qs, 'I(q)': intensities})

    if outputDir is None:
        outputDir = dataDir.parent
    if type(outputDir) == str:
        outputDir = Path(outputDir)
    df_saxs.to_pickle(outputDir / 'saxs.pkl') # Don't save in csv b/c the export cannot handle arrays in a cell
    print('Saved the processed data in %s' % (outputDir / 'saxs.pickle'))
    return df_saxs
def search_cfg_file(dataDir):
    '''
    Searches for the .cfg file in the dataDir
    Returns the path to the .cfg file
    '''
    cfg_files = []
    for file in os.listdir(dataDir):
        if file.endswith('.cfg'):
            cfg_files.append(os.path.join(dataDir, file))
    if len(cfg_files) > 1:
        print('Multiple .cfg files found in the directory')
        for file in cfg_files:
            print(file)
        return cfg_files
    else:
        cfg_file = cfg_files[0]
        return cfg_file


## Handle pickle files
def write_pickle(filepath, obj, verbose=True):
    """
    Generate a pickle file from obj
    Parameters
    ----------
    obj
    filepath
    verbose

    Returns
    -------

    """
    # Extract the directory and filename from the given path
    directory, filename = os.path.split(filepath)[0], os.path.split(filepath)[1]
    if directory == '':
        directory = '.'

    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    pickle_out = open(filepath, "wb")
    pickle.dump(obj, pickle_out)
    if verbose:
        print('Saved data in ' + filepath)
    pickle_out.close()

def read_pickle(filename):
    with open(filename, "rb" ) as pf:
        # Try to read the file with utf-8 encoding
        try:
            obj = pickle.load(pf)
        except UnicodeDecodeError:
            # Try to read the file with byte encoding
            try:
                obj = pickle.load(pf, encoding="bytes")
            # If it fails, try to read the file with pandas
            except:
                obj = pandas.read_pickle(filename)
    return obj

