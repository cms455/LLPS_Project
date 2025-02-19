import argparse
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import os
import pylab
import pickle

####################################################################################################
# User input
pdir = Path(r"Z:\share\tm688\saxs\nmr\complete") # Location of processed NMR data
output_dir = Path(r"Z:\public\saxs\2023-11-03\processed_data") # Location where the csv files will be saved
####################################################################################################

def read_pkl(filename):
    with open(str(filename), 'rb') as f:
        return pickle.load(f)

def create_peg_csv(pdir=Path(r"Z:\share\tm688\saxs\nmr\complete"),
                   output_dir=Path(r"Z:\public\saxs\2023-11-03\processed_data")):
    """
    
    Parameters
    ----------
    pdir: Path, Directory where the extracted nmr data are stored
    ... pdir = Path('/Volumes/Shared/share/tm688/saxs/nmr/complete') # Mac
    ... pdir = Path(r"Z:\share\tm688\saxs\nmr\complete")  # Windows

    Returns
    -------

    """

    # list directories
    seriesNames = ['Dilute', 'Dense']

    # loop through directories
    for seriesName in seriesNames:
        nmrDirList = [x for x in pdir.iterdir() if x.is_dir() and seriesName in x.name]

        # INITIALIZATION
        sampleNo, int_PEG, int_DMF = [], [], []
        conc_PEG_molL, conc_DMF_molL, conc_PEG_gmL, conc_PEG_pctgmL = [], [], [], []
        for nmrDir in nmrDirList:
            fn_dataDirs = sorted(nmrDir.glob('*/'))
            fn_data = fn_dataDirs[-1] / 'data.pkl' # Always use the data from the latest run
            if os.path.exists(fn_data):
                print(f"Series: {seriesName}, Sample: {nmrDir.name}")
                # Grab the nmr data
                nmrData = read_pkl(fn_data)

                sampleNo.append(int(nmrDir.name[6:9]))
                int_PEG.append(nmrData['Integral A'])
                int_DMF.append(nmrData['Integral B'])
                conc_PEG_molL.append(nmrData['Concentration A']) # mol/L
                conc_DMF_molL.append(nmrData['Analysis Params']['concB']) # mol/L
                conc_PEG_gmL.append(nmrData['%w/v A']/100) # weight by volume in g/mL
                conc_PEG_pctgmL.append(nmrData['%w/v A']) # %w/v
        df = pd.DataFrame({'Sample': sampleNo,
                           'Phase': seriesName, # 'Dilute' or 'Dense
                           'Integral_PEG': int_PEG, 'Integral DMF': int_DMF,
                           'conc_PEG_M': conc_PEG_molL, 'conc_DMF_M': conc_DMF_molL,
                            'conc_PEG_gmL': conc_PEG_gmL, 'conc_PEG_pctgmL': conc_PEG_pctgmL})

        print("Quick look on the data:\n", df.head())
        print('-' * 50)
        # fnOut = Path('/Volumes/Shared/share/tm688/saxs/nmr/csv' + seriesName + '.csv') # Mac
        fnOut = output_dir / ("PEGconc" +seriesName + '.csv') # Mac
        df.to_csv(fnOut, index=False)
        print(fnOut, "saved.")

    # Combine the two dataframes
    dfDilute = pd.read_csv(output_dir / "PEGconcDilute.csv")
    dfDense = pd.read_csv(output_dir /  "PEGconcDense.csv")

    dfAll = pd.concat([dfDilute, dfDense])
    fnOut = output_dir / "PEGconcAll.csv"
    dfAll.to_csv(fnOut, index=False)
    print(fnOut, "saved.")

    print(dfAll.head())

def main(pdir=pdir, output_dir=output_dir):
    # Create three csv files: PEGconcDilute.csv, PEGconcDense.csv, PEGconcAll.csv
    ## PEGconcDilute.csv: PEG concentration in dilute phase
    ## PEGconcDense.csv: PEG concentration in dense phase
    ## PEGconcAll.csv: Combined data from PEGconcDilute.csv and PEGconcDense.csv
    create_peg_csv(pdir=pdir, output_dir=output_dir)


if __name__ == '__main__':
    main()