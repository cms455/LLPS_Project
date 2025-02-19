"""
Date: Oct20-2023, Author: Takumi Matsuzawa

This script is used to analyze the 1H NMR data of PEG/DMF mixtures.

The goal is to estimate the concentration of PEG in the mixture.

What this does:
... 1. Read the NMR data from the data server
... 2. Analyze the data to estimate the concentration of PEG
... 3. Save the results in a pickle file (dictionary) and plots a NMR spectrum

How to use:
... Run this script from the command line
... ... DILUTE PHASE: PEG: 133 uL, DMF: 80 uL
... ... ... python .\analyze_nmr.py -i "\\files.cornell.edu\as\CHM\\NMR\Dufresne\tm688\av400" -o "\\en-ms-dufresne-fs01.coecis.cornell.edu\shared\share\tm688\saxs\nmr" -vA 133 -vB 80 -p Dilute_n01_Nov10-2023
... ... DENSE PHASE: PEG: 20 uL, DMF: 80 uL
... ... ... python .\analyze_nmr.py -i "\\files.cornell.edu\as\CHM\\NMR\Dufresne\tm688\av400" -o "\\en-ms-dufresne-fs01.coecis.cornell.edu\shared\share\tm688\saxs\nmr" -vA 20 -vB 80 -p Dense_n01_Nov10-2023

Math:
... [A] = (I_A / I_B) * (Number of 1H in between [X1, X2] (ppm), originating from Species B)
          / (Number of 1H in Species A in between [Y1, Y2] (ppm) PER molecule) * Volume of Species A
... For PEG/DMF mixtures, we have:
        [PEG] = (I_PEG / I_DMF) * (6 [DMF]_stock V_DMF_stock) / (4 * (MW_PEG - 18.02) / 44.05)) / V_PEG
    %w/v concentration of PEG reads
        PEG %w/v = [PEG] * MW_PEG / 10

Chemicals in the samples:
... PEG (Polyethylene glycol): Target molecule whose conc. is quantified
... DMF (N, N-Dimethylformamide): Reference molecule whose conc. is known
... D2O (Deuterium oxide): Solvent- D2O is preferred for clean NMR results because H2O produces a broad peak.
... Extra species such as BSA and H2O can be present for BSA/PEG phase-separated samples. 
... ... These species are not considered in the analysis.

Data Architecture:
The NMR data is stored in the data server maintained by Chemistry Department.
You must be approved to use the NMR facility to access the data.
All Bruker NMR data is stored in the following directory:
... inputDir: /Volumes/Dufresne/NetID/av400, av500 or av501
... dataDir: /Volumes/Dufresne/NetID/av400/Oct20-2023_TM_Sample0123

Important data:
... <datadir>/<exp no>/fid: Raw NMR data (temporal scan) whose FT is the spectrum we are after
... <datadir>/<exp no>/pdata/1/1r: Real part of the FT of the fid file
... ... This is the NMR spectrum we are all familiar with.
... <datadir>/<exp no>/pdata/1/1i: Imaginary part of the FT of the fid file
... <datadir>/<exp no>/pdata/1/peak.xml
... ... MestReNova can be used to view the spectrum and the peak information.

fid, 1r, and 1i files are binary files.
Bruker uses a different convention to store the data in these files.
`DTYPA` is the indicator of the data type used in the fid file.
... DTYPA = 0 means DTYPA = "int", Value = <32bit integer> * 2^NC
... ... NC is the exponent found in the `acqus` file, and changes depending on the experiment.
... DTYPA = 1 means DTYPA = "double", Value = double.

Reference:
    BRUKER Topsoin Acq Commands & Params:
    https://chemistry.as.miami.edu/_assets/pdf/acquistion-parameters.pdf

    Concise guide on how to read off params:
    https://chemistry.wilkes.edu/~trujillo/NMR/How_To.../Parameter_Reference.pdf
"""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pylab

def parse_args():
    """
    Parse command line arguments

    In this script, we primarily assume the following:
        Species A: PEG (Polyethylene glycol)
        Species B: DMF (N, N-Dimethylformamide)
    However, this code can be used for any other species in principle by adjusting nHA and nHB.

    Returns
    -------
    args: argparse.Namespace
    """
    parser = argparse.ArgumentParser(
        description="A script that estimates the concentration of chemical A (e.g. PEG) from a 1H NMR spectrum.")
    parser.add_argument('-i', dest='inputDir', type=str,
                        default='/Volumes/Dufresne/tm688/av400',
                        help='Input directory where NMR data is located')
    parser.add_argument('-p', '--pattern', dest='pattern', type=str,
                        default='',
                        help='pattern of the experiment directory (e.g. Dilute, Dense, etc.)')
    parser.add_argument('-o', dest='outputDir', type=str,
                        default='/Volumes/Shared/share/tm688/sax/nmr',
                        help='Output directory for the analysis output')
    parser.add_argument('-csA', dest='cshiftA', nargs='+', default=[3.3, 3.85],
                        help='Chemical shift for chemical species A (e.g. PEG), default: 3.3, 4.1')
    parser.add_argument('-csB', dest='cshiftB', nargs='+', default=[2.7, 3.2],
                        help='Chemical shift for chemical species B (e.g. DMF) default: 2.7, 3.2')
    parser.add_argument('-nHA', dest='nHA', type=int, default=4,
                        help='Number of protons responsible for the domain between cshiftA `-csA`)')
    parser.add_argument('-nHB', dest='nHB', type=int, default=6,
                        help='Number of protons responsible for the domain between cshiftB `-csB`')
    parser.add_argument('-vA', dest='vA', type=float, default=400.,
                        help='Volume of chemical species A (e.g. PEG) in the sample (in uL), default: 133')
    parser.add_argument('-vB', dest='vB', type=float, default=80.,
                        help='Volume of chemical species B (e.g. DMF) in the sample (in uL), default: 80')
    parser.add_argument('-cB', dest='concB', type=float, default=12.92,
                        help='Molarity of chemical species B (in mol/L)... '
                             'Before dilution with chemical species A and D2O')
    parser.add_argument('-mwA', dest='mwA', type=float, default=4000.,
                        help='Molecular weight of Chemical A (in g/mol)`')
    parser.add_argument('-verbose', type=bool, default=False,
                        help='Verbose mode (True or False)')

    args = parser.parse_args()
    return args


def read_acqus(acqusFile):
    """
    Read parameters related to NMR experiments.

    Read the acqus file and return a dictionary of the parameters
    Acquisition Status Parameters: acqus
    ... These are the parameters that were actually used. Do not confuse yourself with `acqu`.

    References:
    BRUKER Topsoin Acq Commands & Params: https://chemistry.as.miami.edu/_assets/pdf/acquistion-parameters.pdf
    Concise guide to read off params: https://chemistry.wilkes.edu/~trujillo/NMR/How_To.../Parameter_Reference.pdf

    Parameters
    ----------
    acqusFile: str, e.g. /Volumes/Dufresne/tm688/av400/Oct20-2023_TM_Sample0123/acqus

    Returns
    -------
    acqus: dict, acquisition parameters
    ... For unknown keys in the acqus file, the key is set to the line number in the acqus file. e.g. `L100`
    """
    file = Path(acqusFile)
    acqus = {}
    with open(file, 'r') as f:
        ln = 1
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith('##'):
                key, value = line.split('=')
                key = key.strip()
                value = value.strip()
                try:
                    value = float(value)
                except:
                    pass

                if key.startswith('##$'):
                    key = key[3:]
                elif key.startswith('##'):
                    key = key[2:]
                elif key.startswith('$$'):
                    key = key[2:]
                acqus[key] = value
            else:
                acqus[f'L{ln:03d}'] = line
            ln += 1
    return acqus


def get_nmr_spectrum(dataDir, verbose=True):
    """
    Returns two arrays (Chemical shifts and intensity values) for the NMR spectrum from the data directory

    FILE ARCHITECTURE:
    ... / <dataDir> / acqus   # Acquisition Status Parameters
    ... / <dataDir> / fid     # Raw data
    ... / <dataDir> / pdata / 1 / 1r  # Processed data (real)
    ... / <dataDir> / pdata / 1 / 1i  # Processed data (imaginary)
    ... / <dataDir> / pdata / 1 / procs  # Processing Status Parameters

    Parameters
    ----------
    dataDir: str, e.g. /Volumes/Dufresne/tm688/av400/Oct20-2023_TM_Sample0123
    verbose: bool, verbose mode (True or False)

    Returns
    -------
    chemShift: np.ndarray, chemical shift (ppm)
    intensity: np.ndarray, intensity (a.u.) of a NMR spectrum

    """
    # Get the acquisition status parameters
    if verbose:
        print("Reading Acquisition Status Parameters (acqus) ...")
    acqus = read_acqus(dataDir / 'acqus')
    if verbose:
        print("...Done")

    # Get the parameters
    NC = acqus['NC']  # Intensity = <32bit integer> * 2^NC
    BF1 = acqus['BF1']  # Basic Frequency (MHz)
    CtrOfSpctrmHz = acqus['O1']  # Center of Spectrum (Hz)
    CtrOfSpctrm = CtrOfSpctrmHz / BF1  # Center of Spectrum (ppm) = CtrOfSpctrm in Hz / (BF1 * 1e6) * 1e6
    SW = acqus['SW']  # Spectrum Width (ppm)
    chemShiftRange = CtrOfSpctrm - SW / 2., CtrOfSpctrm + SW / 2.  # Range of the scan (chemical shfit)
    TD = int(acqus['TD'])  # Number of Data Points
    chemShift = np.linspace(chemShiftRange[0], chemShiftRange[1], TD)

    # Load Data
    file_1r = dataDir / 'pdata' / '1' / '1r'
    file_1i = dataDir / 'pdata' / '1' / '1i'
    if verbose:
        print("Reading Processing Status Parameters (acqus) ...")
    procs = read_acqus(dataDir / 'pdata' / '1' / 'procs')  # Processing Status Parameters
    if verbose:
        print("...Done")
    NC_proc = procs['NC_proc']

    if verbose:
        print("Reading NMR intensities ...")
    # Read Real and Imaginary Parts of the Fourier Transform
    f = open(file_1r, mode="rb")
    real = np.fromfile(f, dtype=np.int32) * 2 ** NC_proc
    f.close()
    f = open(file_1i, mode="rb")
    imag = np.fromfile(f, dtype=np.int32) * 2 ** NC_proc
    f.close()
    if verbose:
        print("...Done")

    # NMR Spectrum
    # Intensity (Power of Fourier Transform)
    intensity = np.sqrt(real ** 2 + imag ** 2)
    intensity = np.sqrt(real ** 2)
    intensity = intensity[::-1]  # Reverse the order of the elements
    # Chemical Shift (in ppm)
    chemShift = np.linspace(chemShiftRange[0], chemShiftRange[1], len(intensity))

    return chemShift, intensity


def integrate_nmr_spectrum(chemShift, intensity, cs0, cs1):
    """
    Integrate the NMR spectrum between chemical shift `cs0` and `cs1`
    Parameters
    ----------
    chemShift: np.array, chemical shift in ppm
    intensity: np.array, intensity of the NMR spectrum (Power)
    cs0: float, chemical shift of the left boundary; cs0 < cs1
    cs1: float, chemical shift of the right boundary; cs0 < cs1

    Returns
    -------
    integral: float, integral of the NMR spectrum between `cs0` and `cs1`

    """
    # Integrate the NMR spectrum between chemical shift `cs0` and `cs1`
    idx = np.where((chemShift >= cs0) & (chemShift <= cs1))[0]
    integral = np.trapz(intensity[idx], chemShift[idx])
    return integral


def scan_files(inputDir, pattern):
    """
    Scan for `fid` files in the input directory

    Ref:
        Bruker Data Format: http://www.nmragenda-leiden.nl/static/Bruker_manual_4.1.4/topspin_pdf/Bruker_NMR_Data_Formats.pdf
        ... Topspin 3 (DTYPA: "int"): <32bit integer> * 2^NC; NC can be found in acqus file
        ... Topspin 4 (DTYPA: "double"): double

    Parameters
    ----------
    inputDir: str, input directory

    Returns
    -------
    files: list, list of files

    """
    p = Path(inputDir)
    files = list(p.glob("**/fid"))  # Raw Data
    files = [str(file) for file in files]
    files = [Path(file) for file in files if pattern in file]
    return files


def compute_integral(chemShift, intensity, cshiftA, cshiftB, verbose=True):
    """
    Computes the integral of the NMR spectrum between chemical shift cshiftA[0] and cshiftA[1],
    and between cshiftB[0] and cshiftB[1],

    Parameters
    ----------
    chemShift: np.array, chemical shift in ppm
    intensity: np.array, intensity of the NMR spectrum (Power)
    cshiftA: tuple, e.g. (4.0, 4.2) for PEG
    cshiftB: tuple, e.g. (3.0, 3.2) for DMF
    verbose: bool, verbose mode (True or False)

    Returns
    -------
    integral_A: float, integral of the NMR spectrum between `cshiftA[0]` and `cshiftA[1]`
    integral_B: float, integral of the NMR spectrum between `cshiftB[0]` and `cshiftB[1]`
    """
    integral_A = integrate_nmr_spectrum(chemShift, intensity, cshiftA[0], cshiftA[1])  # e.g.  PEG- 4 x C-H (sp3), dd
    integral_B = integrate_nmr_spectrum(chemShift, intensity, cshiftB[0], cshiftB[1])  # e.g. DMF- 6 x C-H (sp3), dd
    return integral_A, integral_B


def plot_spectrum(x, y, cshiftA, cshiftB, integral_A, integral_B,
                  num=1, color=None, alpha_fill=0.5, title=''):
    """
    Plots the NMR spectrum and the integration regions

    Parameters
    ----------
    x: np.array, chemical shift in ppm
    y: np.array, intensity of the NMR spectrum (Power)
    cshiftA: tuple, e.g. (4.0, 4.2) for PEG
    cshiftB: tuple, e.g. (3.0, 3.2) for DMF
    integral_A: float, integral of the NMR spectrum between `cshiftA[0]` and `cshiftA[1]`
    integral_B: float, integral of the NMR spectrum between `cshiftB[0]` and `cshiftB[1]`
    num: int, figure number\
    alpha_fill: float, alpha value for the fill_between function
    title: str, title of the figure

    Returns
    -------
    fig: matplotlib.figure.Figure, figure object
    ax: matplotlib.axes._subplots.AxesSubplot, axes object
    """
    __fontsize__ = 11
    __figsize__ = (7.54, 7.54)
    cmap = 'magma'

    # See all available arguments in matplotlibrc
    params = {'figure.figsize': __figsize__,
              'font.size': __fontsize__,  # text
              'legend.fontsize': __fontsize__,  # legend
              'axes.labelsize': __fontsize__,  # axes
              'axes.titlesize': __fontsize__,
              'xtick.labelsize': __fontsize__,  # tick
              'ytick.labelsize': __fontsize__,
              'lines.linewidth': 2.5}
    pylab.rcParams.update(params)

    fig, ax = plt.subplots(figsize=__figsize__, num=num)
    ax.plot(x, y, color=color)
    idx = np.where((x >= cshiftA[0]) & (x <= cshiftA[1]))[0]
    ax.fill_between(x[idx], y[idx], np.zeros_like(y[idx]), color='C1', alpha=alpha_fill,
                    label='$I_{PEG}$ = ' + f'{integral_A:.2f} a.u.')
    idx = np.where((x >= cshiftB[0]) & (x <= cshiftB[1]))[0]
    ax.fill_between(x[idx], y[idx], np.zeros_like(y[idx]), color='C2', alpha=alpha_fill,
                    label='$I_{DMF}$ = ' + f'{integral_B:.2f} a.u.')
    ax.invert_xaxis()
    ax.set_xlabel("Chemical Shift (ppm)")
    ax.set_ylabel("Intensity (a.u.)")
    ax.set_ylim(0, None)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title(title)

    return fig, ax


def save(path, ext='pdf', close=False, verbose=True, fignum=None, dpi=None, overwrite=True, tight_layout=False,
         savedata=True, transparent=True, bkgcolor='w', **kwargs):
    """
    Save a figure from pyplot

    Parameters
    ----------
    path: string
        The path (and filename, without the extension) to save the
        figure to.
    ext: string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    ext : string (default='png')
        The file extension. This must be supported by the active
        matplotlib backend (see matplotlib.backends module).  Most
        backends support 'png', 'pdf', 'ps', 'eps', and 'svg'.
    close : boolean (default=True)
        Whether to close the figure after saving.  If you want to save
        the figure multiple times (e.g., to multiple formats), you
        should NOT close it in between saves or you will have to
        re-plot it.
    verbose : boolean (default=True)
        Whether to print information about when and where the sss
        has been saved.
    fignum: int or None
    dpi: int or None
    overwrite: bool, overwrite option
    tight_layout: bool, tight_layout option
    transparent: bool, transparent option
    bkgcolor: str, background color
    kwargs: dict, additional arguments

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
    # If the directory does not exist, create it
    if not os.path.exists(directory):
        os.makedirs(directory)

    # path where the figure is saved
    savepath = os.path.join(directory, filename)
    if verbose:
        print(("Saving figure to '%s'..." % savepath))

    # Save the figure
    if transparent:
        plt.savefig(savepath, dpi=dpi, transparent=transparent, **kwargs)
    else:
        plt.savefig(savepath, dpi=dpi, facecolor=bkgcolor, **kwargs)

    # Close it
    if close:
        plt.close(fignum)

    if verbose:
        print("... Done")


def compute_concentration(integral_A, integral_B, conc_B, mw_A, vA, vB, nHA, nHB):
    """
    Compute the concentration of A
    Parameters
    ----------
    integral_A: float, integral of A
    integral_B: float, integral of B
    conc_B: float, concentration of B
    mw_A: float, molecular weight of A
    vA: float, volume of A
    vB: float, volume of B

    Returns
    -------
    conc_A: float, concentration of A

    """
    conc_A = (integral_A / integral_B) * (nHB * conc_B * vB) / (nHA * (mw_A - 18.02) / 44.05 * vA)
    return conc_A


def write_pickle(filename, data):
    """
    Write data to a pickle file

    Parameters
    ----------
    filename: Path, path to the pickle file
    data: dict, data to be saved

    Returns
    -------
    None
    """
    if not filename.parent.exists():
        os.makedirs(filename.parent)

    # open a file, where you ant to store the data
    file = open(filename, 'wb')

    # dump information to that file
    pickle.dump(data, file)

    # close the file
    file.close()

    print('Data saved to', filename)


def log(args, file):
    """
    Export parser arguments to a text file

    Parameters
    ----------
    args: argparse.Namespace, arguments

    Returns
    -------
    None
    """
    filename = Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'params.txt'

    with open(filename, 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        for arg in vars(args):
            f.write(f'{arg}: {getattr(args, arg)}\n')

# Operations
def process_nmr_data(file, args):
    """
    Process a single NMR data file (/.../`fid`)
    ... 1. Integrate the NMR spectrum between `cshiftA` and `cshiftB`
    ... 2. Estimate the concentration of A (e.g. PEG) from the integral values
    ... 3. Save the results in a pickle file (dictionary)
    ... 4. Save the plots in the output directory
    ... 5. Log the arguments in a text file

    Parameters
    ----------
    file: Path, path to the fid file
    args: argparse.Namespace, arguments

    Returns
    -------

    """
    print('=' * 40)
    chemShift, intensity = get_nmr_spectrum(file.parent, verbose=args.verbose)
    integral_A, integral_B = compute_integral(chemShift, intensity, args.cshiftA, args.cshiftB,
                                              verbose=args.verbose)
    # Molarity of A (mol/L)
    conc_A = compute_concentration(integral_A, integral_B, args.concB, args.mwA, args.vA, args.vB, args.nHA,
                                   args.nHB)
    wvPct = conc_A * args.mwA / 10.
    print(f"Concentration of A: {conc_A:.3f} mol/L")
    print(f"%w/v of A: {wvPct:.1f} %")

    datadict = {"Experiment Name": file.parent.parent.name,
                "Experiment No.": file.parent.name,
                "Integral A": integral_A,
                "Integral B": integral_B,
                "Concentration A": conc_A,
                "concA": conc_A,  # for consistency
                "%w/v A": wvPct,
                "chemShift": chemShift,
                "intensity": intensity,
                "Analysis Params": vars(args),
                }

    fileOut_data = Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'data.pkl'
    write_pickle(fileOut_data, datadict)

    # Plot
    fig, ax = plot_spectrum(chemShift, intensity,
                            args.cshiftA, args.cshiftB,
                            integral_A, integral_B,
                            num=1, alpha_fill=0.5,
                            title=file.parent.parent.name + '_' + file.parent.name
                                  + f'\n %w/v PEG: {wvPct:.1f}%, [PEG]= {conc_A:.3f} M"')

    # Save
    ## A figure with chemical shift 0 - 14 ppm
    fileOut_img = Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'spectrum'
    save(fileOut_img, ext='png', dpi=300, transparent=False, bkgcolor='w',
         close=False)

    ## A close-up figure
    ax.set_xlim(4.2, 2.6)
    save(Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'spectrum_closeup', ext='png',
         dpi=300, transparent=False, bkgcolor='w',
         close=False)

    ## A close-up on the PEG peaks
    ax.set_xlim(args.cshiftA[1] + 0.1, args.cshiftA[0] - 0.1)
    ax.set_ylim(0, 5e4)
    save(Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'spectrum_closeup_PEG', ext='png',
         dpi=300, transparent=False, bkgcolor='w',
         close=False)

    ## A close-up on the DMF peaks
    ax.set_xlim(args.cshiftB[1] + 0.1, args.cshiftB[0] - 0.1)
    ax.set_ylim(0, 2.5e6)
    save(Path(args.outputDir) / file.parent.parent.name / file.parent.name / 'spectrum_closeup_DMF', ext='png',
         dpi=300,
         transparent=False, bkgcolor='w',
         close=True)
    log(args, file)

def main():
    """
    Main function

    Returns
    -------
    None
    """
    args = parse_args()
    args.cshiftA = [float(x) for x in args.cshiftA]
    args.cshiftB = [float(x) for x in args.cshiftB]
    files = scan_files(args.inputDir, args.pattern)  # scans for raw data files (not FT data files)

    for file in files:
        process_nmr_data(file, args)

if __name__ == "__main__":
    main()
