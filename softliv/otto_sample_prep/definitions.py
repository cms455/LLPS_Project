import numpy as np
import os
import sys
import datetime
import pickle
from pathlib import Path
import csv
import inspect
import pandas as pd
import shutil



class Sample:
    def __init__(self):
        self.name = ""
        self.sample_id = 0
        self.composition = {}
        self.order_mixing = []
        self.prepared = 0
        self.plate_number = 0
        self.well = ""
        self.stock_concs = {}
        self.sample_volume = 0
        self.recipe = {}


# Set the master directory to work in
master_dir = Path(os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))))
samples_dir = master_dir / 'samples'


def user_input(str):
    return input(str)

def max_concs(stock_dict,chemical_list,composition,fixed_chems):
    '''
    This function calculates the maximum concentrations of the chemicals in the sample given the stock concentrations and the composition of the sample.
    Inputs:
        stock_dict: dictionary containing the stock concentrations of the chemicals
        chemical_list: list of chemicals in the sample
        composition: dictionary containing the composition of the sample
        fixed_chems: list of chemicals that are fixed in the sample and cannot be varied
    Output:
        max_concs: dictionary containing the maximum concentrations of the chemicals in the sample
    '''
    max_concs = {}
    V = 1000
    vol = 0
    for chem in fixed_chems:
        vol += composition[chem]*V/stock_dict[chem]
    remainder_chems = [chem for chem in chemical_list if chem not in fixed_chems]
    for chem in remainder_chems:
        max_concs[chem] = (V - vol)*stock_dict[chem]/V
    return max_concs


def make_sample_from_csv(file,chemical_viscosity):
    '''
    This function reads the csv file and creates a sample object
    The first row of the csv file should be the header which contains the name of the chemical used. 
    The first column should contain the sample ID
    The columns should contain the different concentrations of the chemicals in the sample
    The order from left to right should be the order in which the chemicals will be mixed

    Input: 
    file: The path to the csv file

    Output:
    sample_list: A list of sample objects to be prepared by OT2 
    '''
    sample_list = []
    with open(file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        chemicals = header[1:]
        print(chemicals)
        for row in reader:
            sample = Sample()
            sample.sample_id = int(row[0])
            sample.composition = {chemicals[i]: float(row[i+1]) for i in range(len(chemicals))}
            sample.order_of_mixing = chemicals
            sample_list.append(sample)
    name_list = user_input('what do you want to call the sample list?')
    comments = user_input('any comments on the samples?')
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = [name_list,time,comments]
    filename = 'sample_list-' + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + '.pkl'
    if not os.path.exists(samples_dir):
        os.makedirs('samples')
    with open(samples_dir / filename, 'wb') as f:
        pickle.dump([metadata,chemical_viscosity,sample_list], f)

    print('sample list saved as {}'.format(filename))



def make_sample_list_manually(chemical_viscosity,stock_dict):
    '''
    This function makes a list of samples with variation of a given chemical.
    This is mainly if the sample list is pretty simple to make and involves only variation of one chemical while keeping the rest constant.
    The user has to input various details of the samples such as the chemicals, their composition, etc and the sample list is saved as a pickle file.

    Inputs:
    User inputs the details of the sample list such as the chemicals, their composition, etc.

    '''
    sample_list = []
    print('list of chemicals: {}'.format(chemical_viscosity.keys()))
    while True:  
        ans = user_input('make default sample containing bsa,peg,kp,kcl? (y/n)')
        if ans == 'y':          
            sample_chem_list = ['bsa','peg','kp','kcl']
            sample_conc_list = []
            var = user_input('chemical for variation {}').format(chemical_viscosity.keys()).lower().replace(' ','')          # all chemicals are in lowercases and without spaces

            if var in sample_chem_list:
                sample_chem_list.remove(var)
            
            for chem in sample_chem_list:
                    sample_conc_list.append(float(user_input('set conc of {} in mM or g/L (for bsa and peg)'.format(chem))))
            
            
            while True:
                var_max_conc = list(max_concs(stock_dict,chemical_list=sample_chem_list+[var],composition=dict(zip(sample_chem_list,sample_conc_list)),fixed_chems=sample_chem_list).values())
                str_list = user_input('list of concs for variation of {} in mM or g/L (for bsa and peg)'.format(var)).split()
                conc_list = [float(i) for i in str_list]
                max_conc = max(conc_list)
                if all(max_conc <= x for x in var_max_conc):     
                    break
                else:
                    print('max conc allowed is {}'.format(var_max_conc))
                    continue
            sample_chem_list.append(var)
            
            while True:
                str_list = user_input('list the order of mixing the chemicals in your sample - {}'.format(sample_chem_list+['water'])).split() 
                order_mixing = [i.lower().replace(' ','') for i in str_list]
                if set(order_mixing).issubset(sample_chem_list+['water']) and len(order_mixing) == len(sample_chem_list)+1:
                    break
                else:
                    print('invalid order of mixing')
                    continue
            

            
        elif ans == 'n':
            str_list = user_input('enter list of set chemicals from chemical list {}'.format(chemical_viscosity.keys())).split()
            sample_chem_list = [i.lower().replace(' ','') for i in str_list]
            sample_conc_list = []
            for chem in sample_chem_list:
                sample_conc_list.append(float(user_input('set conc of {} in mM or g/L (for bsa and peg)'.format(chem))))
            var = user_input('chemical for variation from {}').format(chemical_viscosity.keys()).lower().replace(' ','')
            str_list = user_input('list of concs for variation of {} in mM or g/L (for bsa and peg)'.format(var)).split()
            conc_list = [float(i) for i in str_list]

        else:
            print('invalid input')
            continue

        for conc in conc_list:
            sample = Sample()
            sample.composition = dict(zip(sample_chem_list,sample_conc_list))
            sample.composition[var] = conc
            sample.composition['water'] = 0
            sample.order_mixing = order_mixing
            sample_list.append(sample)

        if_bckgr = user_input('do you want to make background samples? (y/n)')
        if if_bckgr == 'y':
            chem_bckgr = user_input('which chemical do you want to make background samples for?').lower().replace(' ','')
            for conc in conc_list:
                sample = Sample()
                sample.composition = dict(zip(sample_chem_list,sample_conc_list))
                sample.composition[chem_bckgr] = 0
                sample.composition['water'] = 0
                sample.order_mixing = order_mixing
                sample_list.append(sample)

        
        for sample in sample_list:
            print(sample.composition)
        print('order of mixing: {}'.format(sample_list[0].order_mixing))
        
        if user_input('confirm sample list? (y/n)') == 'y':
            break
        else:
            if user_input('restart sample list? (y/n)') == 'y':
                print('\n\n******* restarting sample list *******\n\n') 
                sample_list = []
            else:
                return
    
    name_list = user_input('what do you want to call the sample list?')
    comments = user_input('any comments on the samples?')
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    metadata = [name_list,time,comments]
    filename = 'sample_list-' + name_list + '-' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '.pkl'
    if not os.path.exists(samples_dir):
        os.makedirs('samples')
    with open(samples_dir / filename, 'wb') as f:
        pickle.dump([metadata,chemical_viscosity,sample_list], f)

    print('sample list saved as {}'.format(filename))
    

def view_sample_lists(dir = samples_dir, show_recipe=False):
    '''
    This function lists all the sample lists that have been saved in the samples directory.
    Input:
        dir: directory where the sample lists are saved - by default this is going to be master_dir/samples but can be changed if the sample lists are saved in a different directory.
    
    '''
    sample_list = []
    cnt = 0
    for file in sorted(os.listdir(dir)):
        if file.startswith('sample_list'):
            with open(dir / file, 'rb') as f:
                metadata, chemical_viscosity, samples = pickle.load(f)
                print('Sample list {}:'.format(cnt))
                print('sample list: {}'.format(metadata[0]))
                print('time: {}'.format(metadata[1]))
                print('comments: {}'.format(metadata[2]))
                for sample in samples:
                    print(sample.composition)
                    if show_recipe:
                        print(sample.recipe)
                print('order of mixing: {}'.format(samples[0].order_mixing))
                # print('recipe for sample 0: {}'.format(samples[0].recipe))
                print('\n\n')
                cnt += 1
    return


def remove_sample_list(sample_list_nos,dir = samples_dir):
    '''
    This function removes the sample lists that are no longer required.
    Input:
        sample_list_nos: list of sample list numbers that need to be removed
        dir: directory where the sample lists are saved - by default this is going to be master_dir/samples but can be changed if the sample lists are saved in a different directory.
    '''
    sample_lists = sorted([f for f in os.listdir(dir) if f.startswith('sample_list')])
    for ind in sample_list_nos:
        file = sample_lists[ind]
        os.remove(dir / file)
        print('removed {}'.format(file))


def make_recipe(sample_volume,sample,stock_dict):
    '''
    This function makes a recipe for a given sample given the composition and sample volume.
    Inputs:
        sample_volume: volume of the sample in ul
        sample: a sample object which contains the composition of the sample
        stock_dict: dictionary containing the stock concentrations of the chemicals
    
    Output:
        recipe: a dictionary containing the volumes of the chemicals and water that need to be added to the sample to make it.
    '''
    recipe = {}
    for key in sample.composition:
        if key == 'water':
            continue
        recipe[key] = sample_volume * sample.composition[key] / stock_dict[key]
    recipe['water'] = sample_volume - sum(recipe.values())
    return recipe


def num_tips_required(pipette_1,pipette_2,vols):
    '''
    Estimates the number of tips that are required by pipettes for a given list of volumes to be pipetted.
    Inputs:
        pipette_1: list of min and max volumes for smaller pipette 1
        pipette_2: list of min and max volumes for pipette 2
        vols: list of volumes to be pipetted
    '''

    tips = np.zeros(2)
    for vol in vols:
        if vol < pipette_2[0]:
            tips[0] += np.ceil(vol / pipette_1[1])
        elif vol > pipette_2[1]:
            tips[1] += np.ceil(vol/pipette_2[1])
        else:
            tips[1] += 1
        
    return tips    


def write_run_csv(n_plates,pipettes,tip_boxes,stock_dict,stock_vols):
    '''
    Writes a csv file that contains the run details for the OT2.
    The csv file includes all sorts of details such as the pipettes, tip boxes, stock concentrations, etc. that have to be specified before the run.
    '''
    
    tip_names = {'p20':'opentrons_96_tiprack_20ul','p300':'opentrons_96_tiprack_300ul','p1000':'opentrons_96_tiprack_1000ul'}
    pipette_pos = ['left','right']
    dec_loc = 11

    falcon15_wells = ['A1','A2','B1','B2','C1','C2']
    falcon50_wells = ['A3','A4','B3','B4']
    with open('labware_configuration.csv','w',newline='') as f:
        fwriter = csv.writer(f,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        fwriter.writerow(['plate_list','plate_name','deck_location','starting well position'])
        for i in range(n_plates):
            dec_loc -= 1
            fwriter.writerow(['plate','thermoscientificnunc_96_wellplate_2000ul',str(dec_loc),'A1'])
        fwriter.writerow([])

        fwriter.writerow(['pipette_list','name','location (right/left)'])
        for ind,pipette in enumerate(pipettes):
            if pipette == 'p20':
                fwriter.writerow(['pipette','p20_single_gen2',pipette_pos[ind]])
            elif pipette == 'p300':
                fwriter.writerow(['pipette','p300_single_gen2',pipette_pos[ind]])
            elif pipette == 'p1000':
                fwriter.writerow(['pipette','p1000_single_gen2',pipette_pos[ind]])
            fwriter.writerow(['tip box','name','deck location','starting tip position'])
        
            for i in range(int(tip_boxes[ind])):
                dec_loc -= 1
                fwriter.writerow(['tiprack',tip_names[pipette],str(dec_loc),'A1'])  
            fwriter.writerow([]) 
        dec_loc -= 1
        fwriter.writerow([])
        fwriter.writerow(['stock_list','chemical name','concentration','volume (mL)','labware_name','deck_location','location_on_labware','tube_type'])
        for key in stock_dict:
            if len(falcon15_wells) == 0 or len(falcon50_wells) == 0:
                falcon15_wells = ['A1','A2','B1','B2','C1','C2']
                falcon50_wells = ['A3','A4','B3','B4']
                dec_loc -= 1

            if stock_vols[key]:
                if stock_vols[key]/1000 < 8.0:
                    vol_stock = 10
                    tube = 'falcon15'
                    loc = falcon15_wells.pop(0)
                elif stock_vols[key]/1000 < 30.0:
                    vol_stock = 30
                    tube = 'falcon50'
                    loc = falcon50_wells.pop(0)
                else:
                    vol_stock = 50
                    tube = 'falcon50'
                    loc = falcon50_wells.pop(0)
                fwriter.writerow(['stock',key,stock_dict[key],vol_stock,'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical',dec_loc,loc,tube])


def estimate_requirements(sample_volume,sample_list_nos,stock_dict,show=True,file_header='sample_list'):
    '''
    Estimates the requirements for the run such as the pipettes, tip boxes, and stock volumes required based on the sample list and stock concentrations.
    It is useful to run this before running the actual sample prep code to get an idea of the requirements.
    '''
    
    p20_req = 0
    tips_required = []
    all_sample_lists = sorted([f for f in os.listdir(samples_dir) if f.startswith(file_header)])
    vols = []
    n_samples = 0
    stock_vols = dict.fromkeys(stock_dict, 0)
    for ind in sample_list_nos:
        with open(samples_dir / all_sample_lists[ind], 'rb') as f:
            metadata, chemical_viscosity, samples = pickle.load(f)
            for sample in samples:
                n_samples += 1
                recipe = make_recipe(sample_volume,sample,stock_dict)
                vols += list(recipe.values())
                for key in recipe:
                    stock_vols[key] += recipe[key]
                sample.recipe = recipe
                sample.sample_volume = sample_volume
                sample.stock_concs = stock_dict
                if any(vol < 0 for vol in list(recipe.values())):
                    print('error: negative volume for sample {}'.format(sample.composition))
                    print('sample recipe: {}'.format(sample.recipe))
                    return

        
        with open(samples_dir / all_sample_lists[ind], 'wb') as f:
            pickle.dump([metadata,chemical_viscosity,samples], f)


    n_plates = np.ceil(n_samples/96)
    
    if any(vol < 0 for vol in vols):
        print('error: negative volume')

        return

    if any(vol > 0 and vol < 20 for vol in vols):
        p20_req = 1

    if p20_req:
        tips_20_1000 = num_tips_required([0,20],[100,1000],vols)
        tips_20_300 = num_tips_required([0,20],[20,300],vols)
        if tips_20_1000[0] + tips_20_1000[1] < tips_20_300[0] + tips_20_300[1]:
            pipettes = ['p20','p1000']
            tip_boxes = [np.ceil(tips_20_1000[0]/96),np.ceil(tips_20_1000[1]/96)]
            Num_tips = [tips_20_1000[0],tips_20_1000[1]]
        else:
            pipettes = ['p20','p300']
            tip_boxes = [np.ceil(tips_20_300[0]/96),np.ceil(tips_20_300[1]/96)]
            Num_tips = [tips_20_300[0],tips_20_300[1]]
    else:
        tips = [num_tips_required([0,20],[20,300],vols),num_tips_required([0,20],[100,1000],vols),num_tips_required([20,300],[300,1000],vols)]
        total_tips = np.array([tips[0][0]+tips[0][1],tips[1][0]+tips[1][1],tips[2][0]+tips[2][1]])
        print(total_tips)
        min_tip = np.argmin(total_tips)
        if min_tip == 0:
            pipettes = ['p20','p300']
            tip_boxes = [np.ceil(tips[0][0]/96),np.ceil(tips[0][1]/96)]
            Num_tips = [tips[0][0],tips[0][1]]
        elif min_tip == 1:
            pipettes = ['p20','p1000']
            tip_boxes = [np.ceil(tips[1][0]/96),np.ceil(tips[1][1]/96)]
            Num_tips = [tips[1][0],tips[1][1]]
        else:
            pipettes = ['p300','p1000']
            tip_boxes = [np.ceil(tips[2][0]/96),np.ceil(tips[2][1]/96)]
            Num_tips = [tips[2][0],tips[2][1]]
            
    if show == False:
        return int(n_plates),pipettes,tip_boxes,stock_vols
    else:
        print('summary of the minimum requirements:')
        print('number of plates: {}'.format(n_plates))
        print('pipettes: {}'.format(pipettes))
        print('tip boxes: {}'.format(tip_boxes))
        print('Number of tips used: {}'.format(Num_tips))
        print('minimum stock volumes required:')
        for key in stock_vols:
            print('{}: {} mL'.format(key,stock_vols[key]/1000))


 



def create_run_file(sample_volume,sample_list_nos,stock_dict):
    '''
    This function creates a run file for the OT2 based on the sample list and stock concentrations.
    The run file includes the pipettes, tip boxes, stock concentrations, and sample lists that are required for the run.
    '''
    n_plates,pipettes, tip_boxes, stock_vols = estimate_requirements(sample_volume,sample_list_nos,stock_dict,show=False)
    write_run_csv(n_plates,pipettes,tip_boxes,stock_dict,stock_vols)
    filename = 'run_file-' + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + '.pkl'
    sample_lists = [sorted([f for f in os.listdir(samples_dir) if f.startswith('sample_list')])[i] for i in sample_list_nos]

    with open(samples_dir / sample_lists[0], 'rb') as f:
        metadata, chemical_viscosity, samples = pickle.load(f)
    while(True):
        unmixed_chem_list = user_input('Enter the chemicals that need not be mixed immediately?').split()
        if set(unmixed_chem_list).issubset(stock_dict.keys()):
            break
        else:
            print('invalid input')
            continue

    if not os.path.exists(master_dir / 'runfiles'):
        os.makedirs('runfiles')
    with open(master_dir / 'runfiles' / filename, 'wb') as f:
        pickle.dump([chemical_viscosity,unmixed_chem_list,sample_lists], f)
    print('run file saved as {} in folder \'runfiles\''.format(filename))   



def create_labware_file_for_dilutions(n_plates,pipettes,tip_boxes,stock_dict):
    '''
    Writes a csv file that contains the run details for the OT2.
    The csv file includes all sorts of details such as the pipettes, tip boxes, stock concentrations, etc. that have to be specified before the run.
    '''
    
    tip_names = {'p20':'opentrons_96_tiprack_20ul','p300':'opentrons_96_tiprack_300ul','p1000':'opentrons_96_tiprack_1000ul'}
    pipette_pos = ['left','right']
    dec_loc = 11
    falcon15_wells = ['A1','A2','B1','B2','C1','C2']
    falcon50_wells = ['A3','A4','B3','B4']
    with open('dilution_labware.csv','w',newline='') as f:
        fwriter = csv.writer(f,delimiter=',',quotechar='|',quoting=csv.QUOTE_MINIMAL)
        fwriter.writerow(['Sample_plate','plate_name','deck_location'])
        dec_loc -= 1
        fwriter.writerow(['plate','thermoscientificnunc_96_wellplate_2000ul',str(dec_loc)])
        fwriter.writerow([])
        fwriter.writerow(['Tranfer_plate','plate_name','deck_location'])
        dec_loc -= 1
        fwriter.writerow(['plate','thermoscientificnunc_96_wellplate_2000ul',str(dec_loc)])
        fwriter.writerow([])   
        fwriter.writerow(['Dilution_plates_list','plate_name','deck_location'])
        for i in range(n_plates):
            dec_loc -= 1
            fwriter.writerow(['plate','thermoscientificnunc_96_wellplate_2000ul',str(dec_loc)])
        fwriter.writerow([])
 
        fwriter.writerow(['pipette_list','name','location (right/left)'])
        for ind,pipette in enumerate(pipettes):
            if pipette == 'p20':
                fwriter.writerow(['pipette','p20_single_gen2',pipette_pos[ind]])
            elif pipette == 'p300':
                fwriter.writerow(['pipette','p300_single_gen2',pipette_pos[ind]])
            elif pipette == 'p1000':
                fwriter.writerow(['pipette','p1000_single_gen2',pipette_pos[ind]])
            fwriter.writerow(['tip box','name','deck location','starting tip position'])
        
            for i in range(int(tip_boxes[ind])):
                dec_loc -= 1
                fwriter.writerow(['tiprack',tip_names[pipette],str(dec_loc),'A1'])  
            fwriter.writerow([]) 
        
        fwriter.writerow([])
        fwriter.writerow(['stock_list','chemical name','concentration','volume (mL)','labware_name','deck_location','location_on_labware','tube_type'])
        
        vol_water = stock_dict['water']/1000
        while vol_water > 0:
            dec_loc -= 1
            loc = falcon50_wells.pop(0)
            fwriter.writerow(['stock','water',0,45,'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical',dec_loc,loc,'falcon50'])
            vol_water -= 45
        dec_loc -= 1
        loc = falcon50_wells.pop(0)
        fwriter.writerow(['stock','trash',0,10,'opentrons_10_tuberack_falcon_4x50ml_6x15ml_conical',dec_loc,loc,'falcon50'])



def create_run_file_for_dilutions(sample_list_nos = [],vol_sample_per_dil = 20, dilution_factor = 10, Num_dilutions = 1,stock_dict = {},vol_fraction_dense=0.2):
    all_sample_lists = sorted([f for f in os.listdir(master_dir / 'prepared_samples/') if f.startswith('sample_list')])
    dilution_samples = []
    vols = []
    sample_number = 0
    data_list = []

    sample_lists = [sorted([f for f in os.listdir(master_dir / 'prepared_samples/') if f.startswith('sample_list')])[i] for i in sample_list_nos]

    filename = 'dilutions_run_file-' + datetime.datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + '.pkl'
    with open(master_dir / filename, 'wb') as f:
        pickle.dump([vol_sample_per_dil,Num_dilutions,dilution_factor,sample_lists], f)

    vol_water = 0

    for i in sample_list_nos:
        with open(master_dir / 'prepared_samples' / all_sample_lists[i], 'rb') as f:
            metadata, chemical_viscosity, samples = pickle.load(f)
        
        for ind,sample in enumerate(samples):
            sample.sample_id = ind + 1          # remove this later
            for i in range(Num_dilutions):
                sample_number += 1
                vols.append(vol_sample_per_dil)
                vols.append(vol_sample_per_dil*(dilution_factor-1))
                vol_water += vol_sample_per_dil*(dilution_factor-1)

            vol_dense_phase = (vol_fraction_dense)*sample.sample_volume
            vol_dilute_phase = sample.sample_volume - vol_dense_phase
            transfer_vol = 0.6*vol_dilute_phase
            vols.append(transfer_vol)
    stock_dict['water'] = vol_water
        
    
    if any(vol < 0 for vol in vols):
        print('error: negative volume')
        return
    if any(vol > 0 and vol < 20 for vol in vols):
        p20_req = 1
        tips_20_1000 = num_tips_required([0,20],[100,1000],vols)
        tips_20_300 = num_tips_required([0,20],[20,300],vols)
        if tips_20_1000[0] + tips_20_1000[1] < tips_20_300[0] + tips_20_300[1]:
            pipettes = ['p20','p1000']
            tip_boxes = [np.ceil(tips_20_1000[0]/96),np.ceil(tips_20_1000[1]/96)]
            Num_tips = [tips_20_1000[0],tips_20_1000[1]]
        else:
            pipettes = ['p20','p300']
            tip_boxes = [np.ceil(tips_20_300[0]/96),np.ceil(tips_20_300[1]/96)]
            Num_tips = [tips_20_300[0],tips_20_300[1]]
    else:
        tips = [num_tips_required([0,20],[20,300],vols),num_tips_required([0,20],[100,1000],vols),num_tips_required([20,300],[300,1000],vols)]
        total_tips = np.array([tips[0][0]+tips[0][1],tips[1][0]+tips[1][1],tips[2][0]+tips[2][1]])
        min_tip = np.argmin(np.flip(total_tips))
        if min_tip == 2:
            pipettes = ['p20','p300']
            tip_boxes = [np.ceil(tips[0][0]/96),np.ceil(tips[0][1]/96)]
            Num_tips = [tips[0][0],tips[0][1]]
        elif min_tip == 2:
            pipettes = ['p20','p1000']
            tip_boxes = [np.ceil(tips[1][0]/96),np.ceil(tips[1][1]/96)]
            Num_tips = [tips[1][0],tips[1][1]]
        else:
            pipettes = ['p300','p1000']
            tip_boxes = [np.ceil(tips[2][0]/96),np.ceil(tips[2][1]/96)]
            Num_tips = [tips[2][0],tips[2][1]]
    
    print('summary of the minimum requirements:')
    print('pipettes: {}'.format(pipettes))
    print('tip boxes: {}'.format(tip_boxes))
    print('Number of tips used: {}'.format(Num_tips))

    n_plates_dils = int(np.ceil(sample_number/96))
    create_labware_file_for_dilutions(n_plates_dils,pipettes,tip_boxes,stock_dict)

    print('run file saved as {}'.format(filename))


def add_to_prepared_samples(sample_list_nos):
    '''
    This function adds the sample lists which are prepared by otto to the prepared_samples directory.
    Inputs:
        sample_list_nos: list of sample list numbers that need to be added to the prepared_samples directory
    Function:
    Sample lists corresponding to the input will be deleted from the samples folder and put in the prepared_samples folder.
    '''

    if not os.path.exists(master_dir / 'prepared_samples'):
        os.makedirs(master_dir / 'prepared_samples')

    sample_lists = sorted([f for f in os.listdir(samples_dir) if f.startswith('sample_list')])
    for ind in sample_list_nos:
        file = sample_lists[ind]
        shutil.move(samples_dir / file, master_dir / 'prepared_samples' / file)
        print('moved {}'.format(file))


# import paramiko
# from scp import SCPClient

# def createSSHClient(server, port, user, private_key):
#     client = paramiko.SSHClient()
#     client.load_system_host_keys()
#     client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#     client.connect(server, port, username=user, password=None, pkey=private_key)
#     return client

# def sync_files_to_server(server, port, user, private_key_path = '~/.ssh/ot2_ssh_key'):
#     '''
#     This function syncs the files from the local machine to the server.
#     At the moment, it just removes the existing folder with same name in the server and copies the local folder to the server.

#     Inputs:
#         server: server address
#         port: port number
#         user: username
#         private_key_path: path to the private key file
#     '''
    
#     private_key = paramiko.RSAKey(filename=os.path.expanduser(private_key_path))

#     ssh = createSSHClient(server, port, user, private_key)

#     current_directory = os.getcwd()
#     directory_name = os.path.basename(current_directory)

#     # Remove the directory on the server
#     try:
#         stdin, stdout, stderr = ssh.exec_command(f"rm -r /var/lib/jupyter/notebooks/{directory_name}")
#     except Exception as e:
#         pass

#     # Copy the current directory to the server
#     with SCPClient(ssh.get_transport()) as scp:
#         scp.put(current_directory, f"/var/lib/jupyter/notebooks/{directory_name}", recursive=True)

    ############################################################################################################################################################################


    # Otto Functions


def generate_unique_well_position(sample_number,method='diagonal',starts_with='A1'):
    '''
    Function to generate unique well positions for the samples
    '''
    sample_number = np.mod(sample_number,96)
    sample_number = (ord(starts_with[0])-65)*12 + (int(starts_with[1:])-1) + sample_number  
    if method == 'all':
        row = int(sample_number/12)
        col = sample_number%12
        if col == 0:
            col = 12
            row -= 1
        return chr(65+row)+str(col)
    elif method == 'diagonal':
        row = int(sample_number/12)
        col = sample_number%12
        if col == 0:
            col = 12
            row -= 1
        if row%2:
            return chr(65+row)+str(col)
        else:
            return chr(65+row)+str(13-col)

def VolsMixture(conc_dict,stock_dict,vol_sample):
    '''
    Parameters: conc_dict, vol_sample
    Returns: vol_dict
    Description: Calculates the volumes of each solution to be added to the sample
    '''
    vol_dict = {}
    for key in conc_dict:
        vol_dict[key] = [float(conc_dict[key][0])/float(stock_dict[key])*vol_sample,conc_dict[key][1]]    #Volumes in uL
    #make up remaining volume with water
    vol_dict['water'] = [vol_sample - sum(vol_dict.values()),0]
    return vol_dict


def falcon15ml_VtoH(vol):
    '''
    Parameters: vol in ml for now should be more than 2ml
    Returns: height
    Description: Converts volume to height in falcon 15ml tubes
    '''
    angle = 0.315 # radians
    if vol < 2:
        height = np.power(3*vol/(np.pi*np.tan(angle)**2),1/3)*10    # height in mm
        raise ValueError('Volume too low V = {} should be more than 2ml'.format(vol))
    else:
        vol = vol - 1.8
        height = vol*1000/(np.pi*7.33**2) 
        height += 26.48
    return max(height,2)   # height from the top of the tube

def falcon50ml_VtoH(vol):
    '''
    Parameters: vol in ml does not work for less than 5ml
    Returns: height
    Description: Converts volume to height in falcon 50ml tubes
    '''
    tanTheta = 1
    if vol < 5:
        height = (3*vol*1000/tanTheta**2/np.pi)**(1/3) - 2.4

    else:
        vol = vol - 5
        height = vol*1000/(np.pi*(13.5)**2) 
        height += 16
    return max(height,2)   # height from the top of the tube


def Deepwellplate_VtoH(vol):
    '''
    Parameters: vol in uL
    Returns: height
    Description: Converts volume to height in deep well plates
    '''
    area = np.pi*(7.85/2)**2
    height = vol/area
    return max(height-1,2)   # height from the bottom of the well


def sub_volumes(pipette,vol):
    '''
    Break the volume into subvolumes that can be individually transferred
    Input: volume in ul
    Output: list of sub-volumes in ul that add upto the input volume and can be individually transferred
    '''
    vol_max = pipette.max_volume
    vol_list = []
    if vol > vol_max:
        while vol>vol_max:
            vol_list.append(vol_max)
            vol-=vol_max
        if vol<pipette.min_volume:
            vol_last = (vol_list[-1]+vol)/2
            vol_list[-1]=vol_last
            vol_list.append(vol_last)
        else:
            vol_list.append(vol)
        
    else:
        vol_list.append(vol)
    return vol_list


def aspirate(protocol,pipette,vol,rate,stocks_tuberack,stock_dict_entry,delay_seconds=0.5):
    '''
    Parameters: pipette,vol,rate,stocks_tuberack,stock_dict_entry for the stock solution to be added.
    Description: Aspirates a volume of liquid from the stock solution - when stock is in falcon tubes
    '''
    height_asp = 2
    if stock_dict_entry['Tube'] == 'falcon15':
        height_asp = falcon15ml_VtoH(stock_dict_entry['vol']-vol/1000)
    elif stock_dict_entry['Tube'] == 'falcon50':
        height_asp = falcon50ml_VtoH(stock_dict_entry['vol']-vol/1000)
    stock_dict_entry['vol'] -= vol/1000
    pipette.aspirate(vol,stocks_tuberack[stock_dict_entry['loc']].bottom(z=height_asp),rate=rate)
    protocol.delay(seconds=delay_seconds)
    pipette.move_to(stocks_tuberack[stock_dict_entry['loc']].bottom(z=height_asp+10),speed=15)


def water_transfer_and_mix(protocol,pipettes,vol,vol_well,stocks_tuberack,stock_dict,out_well,basic_mix = False,thorough_mix=False,drop_tip=True):
    '''
    Parameters: pipette,vol,in_well,height_asp,out_well,height_disp,n_mix,vol_mix,height_mix,mix=False
    Description: Transfers a volume of liquid with viscosity similar to water from the stock tube to the well
    '''
    pipette = pipettes[vol>pipettes[0].max_volume]
    if pipette.has_tip == False:
        pipette.pick_up_tip()

    water_blowout_rate = 50
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = water_blowout_rate

    if basic_mix:
        height_disp = Deepwellplate_VtoH(vol_well*1.5)
    else:
        height_disp = 2
        
    for vol_disp in sub_volumes(pipette,vol):
        aspirate(protocol,pipette,vol_disp,rate=1,stocks_tuberack=stocks_tuberack,stock_dict_entry=stock_dict)
        pipette.dispense(vol_disp,out_well.bottom(z=height_disp))
        pipette.blow_out(out_well.bottom(z=Deepwellplate_VtoH(vol_well*1.5)))
    
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate

    if thorough_mix:
        thorough_mix(protocol,pipette,out_well,vol_well)
    elif basic_mix:
        n_mix = 2
        vol_mix = max(0.2*vol_well,pipette.min_volume)
        pipette.mix(repetitions=n_mix,volume=vol_mix,location=out_well.bottom(z=2),rate=5)
        pipette.blow_out(out_well.bottom(z=20))


    if pipette.has_tip == True and drop_tip:
        pipette.drop_tip()



def thorough_mix(protocol,pipettes,out_well,vol_well,purpose='SamplePrep'):
    '''
    This is for thorough mixing of the sample in the well - Not to be used for phase separated samples
    Typeically used for mixing the BSA and other salts in the sample just before the PEG is added to it.

    Parameters:
    pipette: pipette to be used for mixing
    out_well: well to be mixed
    vol_well: volume of the sample currently in the well
    purpose: purpose of the mixing - 'SamplePrep' or 'Dilutions' or 'CriticalPt' - I use different mixing parameters for these two purposes
            for the dilutions, the number of mixing steps is greater and use a larger volume for mixing
    '''
    
    if purpose == 'SamplePrep':
        vol_mix = 0.20*vol_well # always mix 20% of the volume of solution in well
        n_mix_down = 10
        n_mix_up = 10
    elif purpose == 'Dilutions':
        vol_mix = 0.50*vol_well # always mix 50% of the volume of solution in well
        n_mix_down = 15
        n_mix_up = 15
    elif purpose == 'CriticalPt':
        vol_mix = 0.20*vol_well
        n_mix_down = 15
        n_mix_up = 15


    # vol_mix = 200    # for the CP samples
    if vol_mix < pipettes[0].min_volume:
        vol_mix = pipettes[0].min_volume
    
    for p in pipettes:
        if p.has_tip:
            if vol_mix <= p.max_volume and vol_mix >= p.min_volume:
                pipette = p 
                break
            else:
                p.drop_tip()
        else:
            pipette = pipettes[vol_mix>pipettes[1].min_volume]
    
    if pipette.has_tip == False:
        pipette.pick_up_tip()

    height_sample = Deepwellplate_VtoH(vol_well)


    h_mix_down_1 = max(0.50*height_sample,2)
    h_mix_down_2 = max(0.35*height_sample,2)

    for i in range(n_mix_down):
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mix_down_1),rate=3)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mix_down_2),rate=5)

    h_mix_up_1 = max(0.50*height_sample,2)
    h_mix_up_2 = max(1*height_sample,2)
    for i in range(n_mix_up):
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mix_up_1),rate=3)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mix_up_2),rate=5)

    # set blow out rates and blow out
    water_blowout_rate = 30
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = water_blowout_rate
    pipette.blow_out(out_well.bottom(z=h_mix_up_2+2))
    # set blow out rates back to default:
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate
    pipette.move_to(out_well.top(z=4))

    pipette.drop_tip()
    

def MixedPhase_mix(protocol,pipettes,out_well,vol_well):
    '''
    For mixing samples which contain PEG - mixed phase samples

    Input:
    pipettes = list of pipettes in the protocol
    out_well = well to be mixed
    vol_well = volume of the sample currently in the well
    '''

    vol_mix = 0.2*vol_well # always mix 20% of the volume of solution in well

    if vol_mix < pipettes[0].min_volume:
        vol_mix = pipettes[0].min_volume

    for p in pipettes:
        if p.has_tip:
            if 2*vol_mix <= p.max_volume and 2*vol_mix >= p.min_volume:
                pipette = p 
                break
            else:
                p.drop_tip()
        else:
            pipette = pipettes[vol_mix>pipettes[1].min_volume]
    
    if pipette.has_tip == False:
        pipette.pick_up_tip()
#    N_mix_init = 20
    N_mix_init = 10
    N_mix_rot = 15
    N_mix_out = 5

    height_sample = Deepwellplate_VtoH(vol_well)
    h_mix_up = max(0.85*height_sample,2)
    h_mix_down = max(0.75*height_sample,2)

    for i in range(N_mix_init):
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mix_up),rate=2)
        protocol.delay(seconds=0.8)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mix_down),rate=10)
    
    h_mixrot_up_1 = max(0.75*height_sample,2)
    h_mixrot_up_2 = max(0.50*height_sample,2)
    h_mixrot_down_1 = max(0.25*height_sample,2)
    h_mixrot_down_2 = max(0.80*height_sample,2)

    print(pipette,vol_mix)
    for i in range(N_mix_rot):
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mixrot_up_1),rate=2)
        protocol.delay(seconds=0.8)
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mixrot_up_2),rate=2)
        protocol.delay(seconds=0.8)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mixrot_down_1),rate=10) 
        protocol.delay(seconds=0.8)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mixrot_down_2),rate=10)  
        protocol.delay(seconds=1)
    
    peg_blowout_rate = 7
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = peg_blowout_rate
    pipette.blow_out(out_well.bottom(z=h_mix_up))
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate

    pipette.drop_tip()


def CriticalPoint_mix(protocol,pipettes,out_well,vol_well,N_mix = 1):
    '''
    This is for mixing phase separating and one-phase samples close to the critical point
    The difference here is that both the solution added (PEG) and the solution in well (BSA+salts) are viscous and mixing of two viscous liquids takes more effort in mixing.
    Input:
    pipettes = list of pipettes in the protocol
    out_well = well to be mixed
    vol_well = volume of the sample currently in the well
    N_mix = number of times to mix the sample typically 3
    '''

    vol_mix = 0.2*vol_well # always mix 20% of the volume of solution in well
    vol_mix_static = 0.45*vol_well
    if vol_mix < pipettes[0].min_volume:
        vol_mix = pipettes[0].min_volume

    for p in pipettes:
        if p.has_tip:
            if vol_mix <= p.max_volume and vol_mix >= p.min_volume:
                pipette = p 
                break
            else:
                p.drop_tip()
        else:
            pipette = pipettes[vol_mix>pipettes[1].min_volume]
    
    if pipette.has_tip == False:
        pipette.pick_up_tip()
#    N_mix_init = 20
    N_mix_init = 10
    N_mix_rot = 15
    N_mix_out = 5

    height_sample = Deepwellplate_VtoH(vol_well)
    h_mix_up = max(0.9*height_sample,2)
    h_mix_down = max(0.75*height_sample,2)
    
    
    for i in range(N_mix_init):
        pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mix_up),rate=4)
        protocol.delay(seconds=0.8)
        pipette.dispense(vol_mix,location=out_well.bottom(z=h_mix_up),rate=10)
    
    h_mixrot_up_1 = max(0.75*height_sample,2)
    h_mixrot_up_2 = max(0.50*height_sample,2)
    h_mixrot_down_1 = max(0.25*height_sample,2)
    h_mixrot_down_2 = max(0.80*height_sample,2)
    
    n_mix = 0
    while n_mix < N_mix:

        for i in range(N_mix_rot):
            pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mixrot_up_1),rate=5)
            protocol.delay(seconds=0.8)
            pipette.aspirate(vol_mix,location=out_well.bottom(z=h_mixrot_up_2),rate=5)
            protocol.delay(seconds=0.8)
            pipette.dispense(vol_mix,location=out_well.bottom(z=h_mixrot_down_1),rate=10) 
            protocol.delay(seconds=0.8)
            pipette.dispense(vol_mix,location=out_well.bottom(z=h_mixrot_down_2),rate=10)  
            protocol.delay(seconds=1)
            
        for i in range(N_mix_init):
            pipette.aspirate(vol_mix_static,location=out_well.bottom(z=h_mix_up),rate=4)
            protocol.delay(seconds=0.3)
            h_mix_down = max(np.random.uniform(0.25,0.9)*height_sample,2)
            pipette.dispense(vol_mix_static,location=out_well.bottom(z=h_mix_down),rate=10)    
        n_mix += 1

    peg_blowout_rate = 7
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = peg_blowout_rate
    pipette.blow_out(out_well.bottom(z=h_mix_up))
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate

    pipette.drop_tip()


def PEG_transfer_and_mix(protocol,pipettes,vol,stocks_tuberack,stock_dict,out_well,vol_well,mix=True, drop_tip=True):
    '''
    Description: Transfers a volume of liquid with viscosity similar to PEG from the stock tube to the well and mixes the solution in the well.
    Input:
    pipettes = list of pipettes in the protocol
    vol = volume of the liquid to be transferred
    stocks_tuberack = labware where the stock solutions are stored
    stock_dict = dictionary containing the stock solutions
    out_well = well to be mixed
    vol_well = volume of the sample currently in the well
    '''

    pipette = pipettes[vol>pipettes[0].max_volume]
    if pipette.has_tip == False:
        pipette.pick_up_tip()
    peg_blowout_rate = 7
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = peg_blowout_rate

    height_disp = Deepwellplate_VtoH(vol_well)
    # height_disp = 2
    for vol_disp in sub_volumes(pipette,vol):
 #       pipette.aspirate(vol_disp,in_well.bottom(z=height_asp),rate=0.3)
        aspirate(protocol,pipette,vol_disp,rate=0.3,stocks_tuberack=stocks_tuberack,stock_dict_entry=stock_dict,delay_seconds=7)

        protocol.delay(seconds=1) # just checking 
    
        pipette.dispense(vol_disp,out_well.bottom(z=height_disp),rate=0.2)
        protocol.delay(seconds=3)

        pipette.blow_out(out_well.bottom(z=Deepwellplate_VtoH(vol_well)+2))
    
    pipette.flow_rate.blow_out = def_pipette
    # mix the sample 
    if mix:
        # CriticalPoint_mix(protocol,pipettes,out_well,vol_well,N_mix=3)
        MixedPhase_mix(protocol,pipettes,out_well,vol_well)

    protocol.delay(seconds=1) # just checking

    # make sure all the tips are dropped
    if pipette.has_tip == True and drop_tip:
        pipette.drop_tip()



def bsa_transfer_and_mix(protocol,pipettes,vol,stocks_tuberack,stock_dict,out_well,vol_well,mix=True,drop_tip=True):
    '''
    Description: Transfers a volume of liquid with viscosity similar to BSA from the stock tube to the well and mixes the solution in the well.
    Input:
    pipettes = list of pipettes in the protocol
    vol = volume of the liquid to be transferred
    stocks_tuberack = labware where the stock solutions are stored
    stock_dict = dictionary containing the stock solutions
    out_well = well to be mixed
    vol_well = volume of the sample currently in the well
    '''
    pipette = pipettes[vol>pipettes[0].max_volume]
    if pipette.has_tip == False:
        pipette.pick_up_tip()

    bsa_blowout_rate = 30
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = bsa_blowout_rate

    for vol_disp in sub_volumes(pipette,vol):
#        pipette.aspirate(vol_disp,in_well.bottom(z=3),rate=0.8)
        aspirate(protocol,pipette,vol_disp,rate=0.6,stocks_tuberack=stocks_tuberack,stock_dict_entry=stock_dict,delay_seconds=2)
        protocol.delay(seconds=1)     # just checking

        height_disp = max(0.25*Deepwellplate_VtoH(vol_well),2)
        pipette.dispense(vol_disp,out_well.bottom(z=height_disp),rate=0.8)

        pipette.blow_out(out_well.bottom(z=Deepwellplate_VtoH(vol_well)+2))
    
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate

    if mix:
        thorough_mix(protocol,pipettes,out_well,vol_well)

    if pipette.has_tip == True and drop_tip:
        pipette.drop_tip()


# def old_AddEverything(protocol,pipettes,row,plate,stock_dict,stocks_tuberack):
#     # misc_type = row[12]
#     vol_water = float(row[1])
#     vol_kcl = float(row[2])
#     vol_kp = float(row[3])
#     vol_bsa = float(row[4])
#     vol_peg = float(row[5]) 
#     # vol_misc = float(row[6])
#     vol_well = 0
#     well_loc = row[18]

#     if vol_water:
#         vol_well += vol_water
#         water_transfer(protocol,pipettes,vol_water,vol_well,stocks_tuberack,stock_dict['water'],plate[well_loc],basic_mix=False)
#     if vol_kp:
#         vol_well += vol_kp
#         water_transfer(protocol,pipettes,vol_kp,vol_well,stocks_tuberack,stock_dict['kp'],plate[well_loc],basic_mix=True)
#     if vol_kcl:
#         vol_well += vol_kcl
#         water_transfer(protocol,pipettes,vol_kcl,vol_well,stocks_tuberack,stock_dict['kcl'],plate[well_loc],basic_mix=True)
#     # if vol_misc:
#     #     vol_well += vol_misc
#     #     water_transfer(protocol,pipettes,vol_misc,vol_well,stocks_tuberack,stock_dict[misc_type],plate[well_loc],basic_mix=True)
#     #     print(stock_dict[misc_type],misc_type)
#     if vol_bsa:
#         vol_well += vol_bsa
#         bsa_transfer(protocol,pipettes,vol_bsa,stocks_tuberack,stock_dict['bsa'],plate[well_loc],vol_well)
#     if vol_peg:
#         vol_well += vol_peg
#         Final_PEG_transfer(protocol,pipettes,vol_peg,stocks_tuberack,stock_dict['peg'],plate[well_loc],vol_well)
#     else:
#         MixedPhase_mix(protocol,pipettes,plate[well_loc],vol_well)


def add_chemical(protocol,pipettes,stock_dict,out_well,chem_name,chem_type,vol_added,vol_well,basic_mix,drop_tip=True):
    '''
    Function to add a particular chemical in the recipe based on its viscosity (chem_type)
    '''
    if vol_added > 0:
        stock_tuberack = stock_dict[chem_name]['labware']
        if chem_type == 'water':
            water_transfer_and_mix(protocol,pipettes,vol_added,vol_well,stock_tuberack,stock_dict[chem_name],out_well,basic_mix=basic_mix,drop_tip=drop_tip)
        elif chem_type == 'bsa':
            bsa_transfer_and_mix(protocol,pipettes,vol_added,stock_tuberack,stock_dict[chem_name],out_well,vol_well,drop_tip=drop_tip)
        elif chem_type == 'peg':
            PEG_transfer_and_mix(protocol,pipettes,vol_added,stock_tuberack,stock_dict[chem_name],out_well,vol_well,drop_tip=drop_tip)
        else:
            raise ValueError('Chemical type not recognized')


    
def Mix_Everything(protocol,pipettes,sample,out_well,stock_dict,chemical_viscosity):
    '''
    Function to add all the chemicals in the recipe to the sample
    '''
    vol_well = 0
    for ind,chem in enumerate(sample.order_mixing):

        if ind == 0:
            basic_mix = False
        else:
            basic_mix = True
        vol_well += sample.recipe[chem]

        print('adding chemical {} and vol {}'.format(chem,sample.recipe[chem]))
        
        add_chemical(protocol,pipettes,stock_dict,out_well,chem,chemical_viscosity[chem],sample.recipe[chem],vol_well,basic_mix)




def load_labware(protocol,master_dir,labware_file = 'labware_configuration.csv'):
    '''
    Load all the labware required for Otto for the protocol which is specificied in the run.csv file 
    '''
    tipracks_1 = []
    tipracks_2 = []
    starting_tips_1 = []
    starting_tips_2 = []
    pipette_list = ['p20_single_gen2','p300_single_gen2','p1000_single_gen2']
    tiprack_list = ['opentrons_96_tiprack_20ul','opentrons_96_tiprack_300ul','opentrons_96_tiprack_1000ul']
    pipette_cnt = -1
    pipettes = []
    plate = []
    stocks = {}
    stock_tuberacks = []
    stock_index = -1
    
    with open(master_dir / labware_file, newline='') as csvfile:
        labware_reader = csv.reader(csvfile, delimiter=',', quotechar='|')

        for row in labware_reader: 
            if any(x.strip() for x in row):
                if row[0] == "pipette":
                    pipette_cnt += 1
                    name_pipette = row[1]
                    loc_pipette = row[2]
                    if name_pipette in pipette_list and loc_pipette in ['left','right']:
                        pipette = protocol.load_instrument(name_pipette, loc_pipette)
                        pipettes.append(pipette)
                    else:
                        raise ValueError('Pipette not specified correctly')
                if row[0] == "tiprack":
                    name_tiprack = row[1]
                    tiprack_deck = int(row[2])
                    starting_tip = row[3]
                    if name_tiprack in tiprack_list and tiprack_deck in np.arange(11)+1:
                        if pipette_cnt:
                            tipracks_2.append(protocol.load_labware(name_tiprack, tiprack_deck))
                            starting_tips_2.append(starting_tip)
                            if name_pipette == pipette_list[2]:
                                tipracks_2[-1].set_offset(x=0.10, y=1.00, z=0.00)
                        else:
                            tipracks_1.append(protocol.load_labware(name_tiprack, tiprack_deck))
                            starting_tips_1.append(starting_tip)
                    else:
                        raise ValueError('Tiprack not specified correctly')
                if row[0] == "plate":
                    name_plate = row[1]
                    plate_deck = int(row[2])
                    starting_well = row[3]
                    try:
                        plate.append(protocol.load_labware(name_plate, plate_deck))
                    except:
                        raise ValueError('Plate not specified correctly')
                if row[0] == "stock":
                    key = row[1]
                    conc = float(row[2])
                    vol = float(row[3])
                    labware_name = row[4]
                    stock_deck = int(row[5])
                    loc = row[6]
                    tube = row[7]
                    try:
                        stock_tuberacks.append(protocol.load_labware(labware_name, stock_deck))
                        stock_index += 1
                    except:
                        pass
                    print(stock_tuberacks,stock_index)
                    stocks[key] = {}
                    stocks[key]['conc'] = conc
                    stocks[key]['vol'] = vol
                    stocks[key]['deck'] = stock_deck
                    stocks[key]['loc'] = loc
                    stocks[key]['Tube'] = tube
                    stocks[key]['labware'] = stock_tuberacks[stock_index]   
            
        print(tipracks_1,tipracks_2)
        pipettes[0].tip_racks = tipracks_1
        if len(tipracks_1):
            pipettes[0].starting_tip = tipracks_1[0].well(starting_tips_1[0])
        pipettes[1].tip_racks = tipracks_2
        if len(tipracks_2):
            pipettes[1].starting_tip = tipracks_2[0].well(starting_tips_2[0])
        
        # order the pipettes in the order of their volume capacity
        pipettes = sorted(pipettes,key=lambda x: x.max_volume)

        return pipettes,plate,stocks,stock_tuberacks,tipracks_1,tipracks_2,starting_tips_1,starting_tips_2,starting_well



def transfer_well_to_well(protocol,pipettes,vol,source_well, vol_src_well, height_asp, dest_well,rate=1,mix=False,drop_tip=True,sep=False):
    '''
    Function to transfer liquid from one well to another
    Input:
        pipettes = list of pipettes in the protocol
        vol = volume of the liquid to be transferred
        source_well = well from which the liquid is to be transferred
        vol_src_well = volume of the liquid currently in the source well
        height_asp = height from the bottom of the source well from which the liquid is to be aspirated
        dest_well = well to which the liquid is to be transferred
        rate = rate of aspiration and dispensing
        mix = whether to mix the liquid in the destination well
        drop_tip = whether to drop the tip after the transfer
        sep = whether to separate the dense and dilute phase mixtures
    '''
    pipette = pipettes[vol>pipettes[0].max_volume]
    if pipette.has_tip == False:
        pipette.pick_up_tip()

    bsa_blowout_rate = 30
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = bsa_blowout_rate

    
    if sep:
        for vol_disp in sub_volumes(pipette,vol):
            print(sub_volumes(pipette,vol))
            if vol_src_well + vol_disp > 2000:
                height_asp = Deepwellplate_VtoH(vol_src_well - 1.1*vol_disp/2)
                pipette.aspirate(vol_disp/2,source_well.bottom(z=height_asp),rate=rate)
                protocol.delay(seconds=0.5)
                height_asp = Deepwellplate_VtoH(vol_src_well - 1.1*vol_disp)
                pipette.aspirate(vol_disp/2,source_well.bottom(z=height_asp),rate=rate)
                pipette.move_to(source_well.top(z=4))
                pipette.dispense(vol_disp,dest_well.bottom(z=2),rate=rate)
                pipette.blow_out(dest_well.bottom(z=Deepwellplate_VtoH(vol*1.5)))
                vol_src_well -= vol_disp
            else:
                height_asp = Deepwellplate_VtoH(vol_src_well - 1.1*vol_disp)
                pipette.aspirate(vol_disp,source_well.bottom(z=height_asp),rate=rate)
                pipette.move_to(source_well.top(z=4))
                pipette.dispense(vol_disp,dest_well.bottom(z=2),rate=rate)
                vol_src_well -= vol_disp
                pipette.blow_out(dest_well.bottom(z=Deepwellplate_VtoH(vol*1.5)))
    else:
        height_asp = Deepwellplate_VtoH(vol_src_well - 1.1*vol)
        pipette.aspirate(vol,source_well.bottom(z=height_asp),rate=rate)
        protocol.delay(seconds=0.5)
        pipette.move_to(source_well.top(z=4))
        pipette.dispense(vol,dest_well.bottom(z=2),rate=rate)
        pipette.blow_out(dest_well.bottom(z=Deepwellplate_VtoH(vol*1.5)))
    
    if mix:
        thorough_mix(protocol,pipette,dest_well,vol)

    height_blow_out = Deepwellplate_VtoH(vol*1.5)
    pipette.blow_out(dest_well.bottom(z=height_blow_out))
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate  

    if pipette.has_tip == True and drop_tip:
        pipette.drop_tip()


def transfer_well_to_trash(protocol,pipettes,vol,source_well, height_asp, stock_dict):
    '''
    Function to transfer liquid from one well to the trash 
    This is mainly to get rid of any excess dilute phase liquid above the dense phase after separating the two phases
    Input:
        pipettes = list of pipettes in the protocol
        vol = volume of the liquid to be transferred
        source_well = well from which the liquid is to be transferred
        height_asp = height from the bottom of the source well from which the liquid is to be aspirated
        stock_dict = dictionary containing the location of the destination tube
    '''
    pipette = pipettes[vol>pipettes[0].max_volume]
    if pipette.has_tip == False:
        pipette.pick_up_tip()

    bsa_blowout_rate = 30
    def_pipette = pipette.flow_rate.blow_out
    pipette.flow_rate.blow_out = bsa_blowout_rate
    pipette.aspirate(vol,source_well.bottom(z=height_asp),rate=1)
    pipette.move_to(source_well.top(z=4))
    dest_tube = stock_dict['trash']['labware'][stock_dict['trash']['loc']]  

    pipette.dispense(vol,dest_tube.bottom(z=falcon50ml_VtoH(stock_dict['trash']['vol'])),rate=1)
    pipette.blow_out(dest_tube.bottom(z=falcon50ml_VtoH(stock_dict['trash']['vol'])))
    pipette.flow_rate.blow_out = def_pipette #set to default blow out rate
    pipette.drop_tip()
    


def add_mixing_vols(protocol,pipettes,plate,stock_dict,chemical_viscosity,unmixed_chem_list=[],sample_df=None):
    '''
    Function to add all the chemicals in the recipe to the sample
    '''

    output_dir = Path('/var/lib/jupyter/notebooks/')

    for index,row in sample_df.iterrows():
        sample_name = row['Name of sample list']
        sample_number = row['Sample_ID']
        sample_well = row['Well']
        try:
            with open(output_dir / 'logfile.txt', 'a') as f:  # Open the file for writing
                f.write(f'Mixing sample - {sample_name}, sample number - {sample_number} in well - {sample_well} \n')
        except:
            pass

        for ind,chem in enumerate(row['order_mixing']):
            if chem not in unmixed_chem_list:
                if ind == 0:
                    basic_mix = False
                else:
                    basic_mix = True
                vol_well = row['Well Volume'] + row[chem]
                out_well = plate[row['Plate']][row['Well']]
                row['Well Volume'] = vol_well
                add_chemical(protocol,pipettes,stock_dict,out_well,chem,chemical_viscosity[chem],row[chem],vol_well,basic_mix)
        sample_df.at[index,'Well Volume'] = vol_well

    return sample_df



def add_unmixing_vols(protocol,pipettes,plate,stock_dict,chemical_viscosity,unmixed_chem_list=[],sample_df=None):
    '''
    Function to add the volumes of the sample that do not require mixing to the wells
    '''
    
    for chem in unmixed_chem_list:
        chem_vols = np.column_stack((sample_df.index.values, sample_df[chem].values))
        mask = chem_vols[:,1] > pipettes[0].max_volume
        for_pipette_2 = chem_vols[mask]
        for_pipette_1 = chem_vols[~mask]
        
        if for_pipette_1.size > 0:
            pipettes[0].pick_up_tip()
            for index, vol in for_pipette_1:
                out_well = plate[sample_df.loc[index]['Plate']][sample_df.loc[index]['Well']]
                add_chemical(protocol,pipettes,stock_dict,out_well,chem,'water',vol,sample_df.loc[index]['Well Volume'],basic_mix=False,drop_tip=False)
                sample_df.at[index,'Well Volume'] = sample_df.loc[index]['Well Volume'] + vol
            pipettes[0].drop_tip()
        if for_pipette_2.size > 0:
            pipettes[1].pick_up_tip()
            for index,vol in for_pipette_2:
                out_well = plate[sample_df.loc[index]['Plate']][sample_df.loc[index]['Well']]
                add_chemical(protocol,pipettes,stock_dict,out_well,chem,'water',vol,vol_well=sample_df.loc[index]['Well Volume'],basic_mix=False,drop_tip=False)
                sample_df.at[index,'Well Volume'] = sample_df.loc[index]['Well Volume'] + vol
            pipettes[1].drop_tip()

    return sample_df

        
        
