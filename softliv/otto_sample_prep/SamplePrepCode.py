"""
Author: Kaarthik Varma
Date: Nov 14 2023

Code for sample preparation using Otto

Input:
    run.csv file which contains all the labware, pipettes and stocks used for the protocol

"""
import numpy as np
import csv
import argparse
from opentrons import protocol_api
import pickle
from pathlib import Path
import os
import sys
import pandas as pd
import inspect
# src_file_path = inspect.getfile(lambda: None)
current_file_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))

sys.path.append(current_file_directory)
# sys.path.append(os.getcwd())

from definitions import *


metadata = {
        'apiLevel': '2.13',
        'protocolName': 'Beamtime Sample Prep',
        'description': '''Preparing Phase separated samples for the beamtime''',
        'author': 'Kaarthik'
        }

def make_sample_dataframe(sample_lists,start_well = 'A1'):
    df_list = []
    sample_number = 0   
    plate_num = 0
    num_wells_occupied = 0

    for sample_list in sample_lists: 
        with open(samples_dir / sample_list, 'rb') as f:
            metadata_sample, chemical_viscosity, samples = pickle.load(f)

        print('making sample list {}'.format(metadata_sample[0]))
        for sample in samples:
            sample_number += 1
            well_loc = generate_unique_well_position(sample_number=sample_number,method='all',starts_with=start_well)
            num_wells_occupied += 1
            if sample_number >= 96:
                plate_num += 1
                num_wells_occupied = 0
            sample.well = well_loc
            sample.plate_number = plate_num
            sample.sample_id = sample_number
            data = [sample_number, metadata_sample[0], 0,sample.plate_number, sample.well] + list(sample.recipe.values()) + [sample.order_mixing,sample.composition]
            columns = ['Sample_ID', 'Name of sample list', 'Well Volume','Plate','Well'] + list(sample.recipe.keys()) + ['order_mixing','Composition']
            df_sample = pd.DataFrame([data], columns=columns)
            df_list.append(df_sample)
            if not os.path.exists(master_dir / 'prepared_samples'):
                os.makedirs('prepared_samples')
            with open(master_dir / 'prepared_samples' / sample_list,'wb') as f:
                pickle.dump([metadata_sample,chemical_viscosity,samples],f)

    df = pd.concat(df_list, ignore_index=True)
    df = df.fillna(0)

    return df


def run(protocol: protocol_api.ProtocolContext):
    print(current_file_directory)
    # df = pd.DataFrame(columns=['Sample_ID','Plate','Well','Sample Volume','Composition'])
    data_list = []
    # labware definitions:
    pipettes,plates,stock_dict,stock_all_tuberacks,tipracks_1,tipracks_2,starting_tips_1,starting_tips_2,starting_well = load_labware(protocol,master_dir)
    first_run_file = sorted([f for f in os.listdir(master_dir / 'runfiles') if f.startswith('run_file')])[0]
    with open(master_dir / 'runfiles' / first_run_file,'rb') as f:
        chemical_viscosity,unmixed_chem_list,sample_lists = pickle.load(f)

    df = make_sample_dataframe(sample_lists,start_well=starting_well)
    # add all the unmixing solutions:
    df = add_unmixing_vols(protocol,pipettes,plates,stock_dict,chemical_viscosity,unmixed_chem_list,df)
    # add all the mixing solutions:
    df = add_mixing_vols(protocol,pipettes,plates,stock_dict,chemical_viscosity,unmixed_chem_list,df)

    for sample_list in sample_lists:
        os.remove(samples_dir / sample_list)
    os.remove(master_dir / 'runfiles' / first_run_file)

    df_summary = df[['Sample_ID','Name of sample list','Plate','Well','Well Volume','Composition']]

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    summary_file = 'SamplePrepSummary_{}.csv'.format(date)

    if not os.path.exists(master_dir / 'otto_output_folder'):
        os.makedirs('otto_output_folder')
    
    if os.path.exists(master_dir / 'otto_output_folder' / summary_file):
        with open(master_dir / 'otto_output_folder' / summary_file, 'a') as f:
            df.to_csv(f,index=False, header=f.tell()==0)
    else:
        df.to_csv(master_dir / 'otto_output_folder' / summary_file,index=False)
    
