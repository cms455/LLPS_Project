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
        'protocolName': 'Sample Dilutions',
        'description': '''Preparing dilutions of samples''',
        'author': 'Kaarthik'
        }


def convert_sample_lists_to_df(vol_sample_per_dil,Num_dilutions,dilution_factor,sample_lists):
    data_list = []
    sample_number = 0
    sample_id_number = 0
    for sample_list in sample_lists:
        with open(master_dir / 'prepared_samples' / sample_list, 'rb') as f:
            metadata_sample, chemical_viscosity, samples = pickle.load(f)
        for ind,sample in enumerate(samples):
            sample_id_number += 1          # remove this later
            for i in range(Num_dilutions):
                sample_number += 1
                dil_well_loc = generate_unique_well_position(sample_number=sample_number,method='all',starts_with='C1')
                data_list.append([sample_id_number,metadata_sample[0],sample.plate_number,sample.well, sample.sample_volume,sample.composition,dil_well_loc,dilution_factor,Num_dilutions,vol_sample_per_dil,vol_sample_per_dil*(dilution_factor-1)])   
        
    df = pd.DataFrame(data_list,columns=['Sample_ID','Name of Sample','Plate','Well','Sample Volume','Composition','Dilution_Well','Dilution_Factor','Num_Dilutions','SampleVolumeDilution','WaterVolumeDilution'])
    return df
            


def run(protocol: protocol_api.ProtocolContext):
    first_run_file = sorted([f for f in os.listdir(master_dir) if f.startswith('dilutions_run_file')])[0]
    print(first_run_file)
    with open(master_dir / first_run_file,'rb') as f:
        vol_sample_per_dil,Num_dilutions,dilution_factor,sample_lists = pickle.load(f)

    df = convert_sample_lists_to_df(vol_sample_per_dil,Num_dilutions,dilution_factor,sample_lists)
    # labware definitions:
    pipettes,plate,stock_dict,stock_all_tuberacks = load_labware(protocol,master_dir,labware_file='dilution_labware.csv')
    print(pipettes,plate,stock_dict,stock_all_tuberacks)

    sample_plate = plate[0]
    dilution_plate = plate[2]
    transfer_plate = plate[1]



    vol_fraction_dense = 0.2

    # transfer the dilute phase of samples to new plate:
    for sample_id, group in df.groupby('Sample_ID'):
        row = group.iloc[0]
        sample_ID = row['Sample_ID']
        src_well = sample_plate[row['Well']]
        print(src_well)
        tranfer_well = transfer_plate[row['Well']]
        dilution_well = dilution_plate[row['Dilution_Well']]
        sample_vol = row['Sample Volume']
        vol_dense_phase = (vol_fraction_dense)*sample_vol
        vol_dilute_phase = sample_vol - vol_dense_phase
        transfer_vol = 0.7*vol_dilute_phase
        height_asp = Deepwellplate_VtoH(sample_vol-1.1*transfer_vol)
        # transfer_well_to_well(protocol,pipettes,vol=transfer_vol,source_well=src_well,vol_src_well=sample_vol,dest_well=tranfer_well,height_asp=height_asp,drop_tip=False,sep=True)
        height_asp = Deepwellplate_VtoH(vol_dense_phase*1.1)
        remaining_dilute_phase = vol_dilute_phase - transfer_vol
        transfer_well_to_trash(protocol,pipettes,vol=remaining_dilute_phase,source_well=tranfer_well,height_asp=height_asp,stock_dict=stock_dict)


    # # Performing dilutions on the dilute phase

    # # put all the water first:
    # water_vols = np.column_stack((df.index.values, df['WaterVolumeDilution'].values))
    # mask = water_vols[:,1] > pipettes[0].max_volume
    # for_pipette_2 = water_vols[mask]
    # for_pipette_1 = water_vols[~mask]
    # if for_pipette_1.size > 0:
    #     pipettes[0].pick_up_tip()
    #     print('picking up 300 tip here')
    #     for index, vol in for_pipette_1:
    #         out_well = dilution_plate[df.loc[index]['Dilution_Well']]
    #         add_chemical(protocol,pipettes,stock_dict,out_well,'water','water',vol,vol,basic_mix=False,drop_tip=False)
    #     pipettes[0].drop_tip()
    # if for_pipette_2.size > 0:
    #     pipettes[1].pick_up_tip()
    #     for index,vol in for_pipette_2:
    #         out_well = dilution_plate[df.loc[index]['Dilution_Well']]
    #         add_chemical(protocol,pipettes,stock_dict,out_well,'water','water',vol,vol,basic_mix=False,drop_tip=False)
    #     pipettes[1].drop_tip()



    # last_well = None
    # for index, row in df.iterrows():
    #     sample_vol = row['Sample Volume']
    #     vol_dense_phase = (vol_fraction_dense)*sample_vol
    #     vol_dilute_phase = sample_vol - vol_dense_phase
    #     transfer_vol = 0.7*vol_dilute_phase
    #     if last_well != row['Dilution_Well']:
    #         vol_src_well = transfer_vol
    #     tranfer_well = transfer_plate[row['Well']]
    #     dilution_well = dilution_plate[row['Dilution_Well']]
    #     vol_sample_dilution = row['SampleVolumeDilution']
    #     height_asp = Deepwellplate_VtoH(vol_src_well - 1.2*vol_sample_dilution)
    #     vol_dilution_well = vol_sample_dilution + df.loc[index]['WaterVolumeDilution']
    #     print(vol_dilution_well)
    #     transfer_well_to_well(protocol,pipettes,vol=row['SampleVolumeDilution'],source_well=tranfer_well,vol_src_well=vol_src_well,dest_well=dilution_well,height_asp=height_asp,drop_tip=False)
    #     thorough_mix(protocol,pipettes,out_well=dilution_well,vol_well = vol_dilution_well,purpose='Dilutions')
    #     vol_src_well -= vol_sample_dilution
    #     last_well = row['Dilution_Well']
    
    # save the dilution summary file
    date = datetime.datetime.now().strftime("%Y-%m-%d")
    dilution_summary_file = 'DilutionSummary_{}.csv'.format(date)
    df_summary = df[['Sample_ID','Name of Sample','Plate','Well','Dilution_Well','Sample Volume','Composition','Dilution_Factor']]
    if os.path.exists(dilution_summary_file):
        with open(dilution_summary_file, 'a') as f:
            df_summary.to_csv(f,index=False, header=f.tell()==0)
    else:
        df_summary.to_csv(dilution_summary_file,index=False)
    
    # remove the dilution run file:
    # os.remove(master_dir / first_run_file)

