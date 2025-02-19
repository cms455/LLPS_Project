"""
This module contains functions to help your daily lab work.
- make_recipe: This function takes in stock concentrations, final concentrations, dilutions, and sample volumes and returns a recipe
"""


import numpy as np
import pandas as pd
from datetime import datetime

rho_water = 1.0e3 # water mass density, 1000 g/L = 1g/mL
MW_water = 18.01528 # Da = g/mol
molarity_water = rho_water / MW_water  # 55.5 M
def make_solution(volume_mL, molarity_mM, mw_Da, nH2O=0, name='ATP'):
    """
    This function takes in the volume, molarity, molecular weight, and water molarity
    and returns the required amount of solute to make the solution

    Without hydrates, the required number of soltues in mol is
    ... N = volume * molarity # mol = L * mol/L

    With hydrates,
    ... N = volume * adjusted_molarity * hyrate_factor
    ... with hydrate_factor = (molarity_water) / (adjusted_molarity + molarity_water)
    ... This `hydrate_factor` is negligible for most solutions. Most people don't even bother.

    Examples:
        # 1. Make 100mM ATP solution in 1mL of water using ATP-disodium salt trihydrate (Mw=605.2 Da)
        weight_ATP = make_stock(1, 100, 605.2, nH2O=3, name='ATP')
        # 2. Make 100mM ATP solution in 1mL of water using MgCl2 (Mw=95.21 Da)
        weight_ATP = make_stock(1, 100, 605.2, nH2O=3, name='MgCl2')

    Parameters
    ----------
    volume_mL: float, target volume of stock solution in mL
    molarity_mM: float, target molarity of stock solution in M=mol/L
    mw_Da: molecular weight of the solute in Da=g/mol
    nH2O: int, number of hydrates, only relevant if the solvent is water
    name: str, name of the solute (optional), only used for printing

    Returns
    -------
    weight: float, weight of the stock solution in g
    """
    volume, molarity, mw = volume_mL / 1000, molarity_mM / 1000, mw_Da
    # Effect of hydrates
    gamma = molarity_water / (
                molarity_water - molarity)  # [M_water / (M_water - M) ] ; M_water=55,000 mM; mostly this is 1
    adjusted_molarity = molarity * gamma  # M gamma
    N = volume * (adjusted_molarity * molarity_water) / (adjusted_molarity + molarity_water) # with hydrates
    # N = volume * molarity # mol = L * mol/L # without hydrates

    weight = N * mw  # g = mol * (g/mol)
    print(f'Recipe for {molarity_mM:.0f} mM {name} solution:')
    print(f'... Mix {weight:.5f} g = {weight * 1000:.3f} mg in {volume * 1000: .3f} mL of solvent.')
    return weight


def make_recipe(stockConcentrations, finalConcentrations, dilutions, sampleVolumes=100, save=False, filename=None):
    """
    This function takes in stock concentrations, final concentrations, dilutions, and sample volumes and returns a recipe

    Example:
        # stock concentrations
        stockConcentrations = {
            'PEG': 60, # %w/v, (g/100mL)
            'BSA': 26.5, # %w/v, (g/100mL)   265g/L = 26.5g/100mL
            'KCl': 4000, # mM
            'KP': 500, # mM
            'Rhoadmine': 1, # mg/mL,
            'NaHCO3': 1000, #mM
        }

        # final concentrations (µM)
        finalConcentrations = {
            'PEG': 22.6, # %w/v, (g/100mL)
            'BSA': 3.328,# %w/v, (g/100mL)
            'KCl': 200, # mM
            'KP': 100, # mM
            'Rhoadmine': 0.01, # mg/mL
            'NaHCO3': list(range(0, 250, 50)), #mM
        }

        # neccessary dilutions
        dilutions = {
            'PEG': 1,
            'BSA': 1,
            'KCl': 1,
            'KP': 1,
            'Rhoadmine':1,
            'NaHCO3': 1, #mM
        }

    Parameters
    ----------
    stockConcentrations: dict, list of stock concentrations
    finalConcentrations: dict, list of final concentrations
    dilutions: dict, list of dilutions
    sampleVolumes: int, list of sample volumes

    Returns
    -------
    recipe: pandas.DataFrame, a recipe for the given stock concentrations, final concentrations, dilutions, and sample volumes
    """
    # This function takes in stock concentrations, final concentrations, dilutions, and sample volumes and returns a recipe
    def isVariable(variable):
        return isinstance(variable, np.ndarray) or isinstance(variable, list)

    # calculate recipe
    ingredients = list(stockConcentrations.keys())
    ingredientsAndDilutions = [ingredients[i // 2] if i % 2 == 1 else ingredients[i // 2] + ' dilution' for i in
                               range(2 * len(ingredients))]

    # check for variable final concentrations and dilutions
    variableParameters = [key for key, value in finalConcentrations.items() if
                          isinstance(value, np.ndarray) or isinstance(value, list)]
    variableDilutions = [key for key, value in dilutions.items() if
                         isinstance(value, np.ndarray) or isinstance(value, list)]

    recipe = pd.DataFrame(columns=ingredientsAndDilutions + ['Water', 'Total'])

    if len(variableParameters) > 0:
        if not isVariable(sampleVolumes):
            for i in range(len(finalConcentrations[variableParameters[0]])):
                # Create labels for the row based on variable parameters
                labels = [str(round(finalConcentrations[value][i], 3)) + ' ' + str(value) for value in
                          variableParameters]
                label = ' '.join(labels)
                # A sub recipe is a single recipe for a given set of variable parameter values
                subRecipe = [0] * len(ingredientsAndDilutions)
                volume = 0
                for k, ingredient in enumerate(ingredientsAndDilutions):
                    # variable parameter but not variable dilution
                    if ingredient in variableParameters and not ingredient in variableDilutions:
                        subRecipe[k] = sampleVolumes * finalConcentrations[ingredient][i] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient])
                        volume += subRecipe[k]
                    # variable parameter and variable dilution
                    elif ingredient in variableParameters and ingredient in variableDilutions:
                        subRecipe[k] = sampleVolumes * finalConcentrations[ingredient][i] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient][i])
                        volume += subRecipe[k]
                    # dilutions
                    elif 'dilution' in ingredient:
                        if any(ingredient.replace(' dilution', '') == value for value in variableDilutions):
                            subRecipe[k] = dilutions[ingredient.replace(' dilution', '')][i]
                        else:
                            subRecipe[k] = dilutions[ingredient.replace(' dilution', '')]
                    # non variable parameter and dilution
                    elif ingredient not in variableParameters and ingredient not in variableDilutions:
                        subRecipe[k] = sampleVolumes * finalConcentrations[ingredient] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient])
                        volume += subRecipe[k]
                    else:
                        subRecipe[k] = sampleVolumes * finalConcentrations[ingredient] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient][i])
                        volume += subRecipe[k]

                # Calculate the buffer volume needed to reach the correct total volume
                buffer = sampleVolumes - volume
                # Add the sub recipe to the list of recipes
                recipe.loc[label] = subRecipe + [buffer] + [sampleVolumes]
        else:
            for i in range(len(finalConcentrations[variableParameters[0]])):
                # Create labels for the row based on variable parameters
                labels = [str(round(finalConcentrations[value][i], 3)) + ' ' + str(value) for value in
                          variableParameters]
                label = ' '.join(labels)
                # A sub recipe is a single recipe for a given set of variable parameter values
                subRecipe = [0] * len(ingredientsAndDilutions)
                volume = 0
                for k, ingredient in enumerate(ingredientsAndDilutions):
                    # variable parameter but not variable dilution
                    if ingredient in variableParameters and not ingredient in variableDilutions:
                        subRecipe[k] = sampleVolumes[i] * finalConcentrations[ingredient][i] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient])
                        volume += subRecipe[k]
                    # variable parameter and variable dilution
                    elif ingredient in variableParameters and ingredient in variableDilutions:
                        subRecipe[k] = sampleVolumes[i] * finalConcentrations[ingredient][i] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient][i])
                        volume += subRecipe[k]
                    # dilutions
                    elif 'dilution' in ingredient:
                        if any(ingredient.replace(' dilution', '') == value for value in variableDilutions):
                            subRecipe[k] = dilutions[ingredient.replace(' dilution', '')][i]
                        else:
                            subRecipe[k] = dilutions[ingredient.replace(' dilution', '')]
                    # non variable parameter and dilution
                    elif ingredient not in variableParameters and ingredient not in variableDilutions:
                        subRecipe[k] = sampleVolumes[i] * finalConcentrations[ingredient] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient])
                        volume += subRecipe[k]
                    else:
                        subRecipe[k] = sampleVolumes[i] * finalConcentrations[ingredient] / (
                                    stockConcentrations[ingredient] / dilutions[ingredient][i])
                        volume += subRecipe[k]

                # Calculate the buffer volume needed to reach the correct total volume
                buffer = sampleVolumes[i] - volume
                # Add the sub recipe to the list of recipes
                recipe.loc[label] = subRecipe + [buffer] + [sampleVolumes[i]]
    else:
        label = 'uL'
        # A sub recipe is a single recipe
        subRecipe = [0] * len(ingredientsAndDilutions)
        volume = 0
        for k, ingredient in enumerate(ingredientsAndDilutions):
            if 'dilution' in ingredient:
                subRecipe[k] = dilutions[ingredient.replace(' dilution', '')]
            else:
                print(ingredient)
                subRecipe[k] = sampleVolumes * finalConcentrations[ingredient] / (
                            stockConcentrations[ingredient] / dilutions[ingredient])
                volume += subRecipe[k]
        # Calculate the buffer volume needed to reach the correct total volume
        buffer = sampleVolumes - volume
        # Add the sub recipe to the list of recipes
        recipe.loc[label] = subRecipe + [buffer] + [sampleVolumes]

    # Drop columns where all values are 1 for dilution or 0 for others
    recipe = recipe.drop(columns=[col for col in recipe.columns if all(recipe[col] == (1 if 'dilution' in col else 0))])
    # Calculate ingredient totals, excluding 'Total' and any 'dilution' columns
    non_dilution_and_total_columns = [col for col in recipe.columns if 'dilution' not in col and col != 'Total']
    recipe.loc['Total stock', non_dilution_and_total_columns] = recipe[non_dilution_and_total_columns].sum()
    # Transpose, round, and replace NaN values with a blank space
    recipe = recipe.transpose().round(decimals=2).fillna('')

    if save:
        if filename is None:
            filename = './recipe_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.csv'
        recipe.to_csv(filename)
        print('Recipe saved as ' + filename)
    return recipe


# UNIT HELPERS
def pcwv2mM(pcwv, mw):
    """
    Convert a percentage weight/volume, %w/v = % g/mL = g/100mL, (pcwv) to molarity (mM)
    Parameters
    ----------
    pcwv: float, percentage weight/volume, %w/v = % g/mL = g/100mL
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    molarity_mM: float, molarity in mM
    """
    M = pcwv * 10 / mw
    mM = M * 1000 # mM = pcwv / mw * 10,000
    return mM

def pcwv2M(pcwv, mw):
    """
    Convert a percentage weight/volume, %w/v = % g/mL = g/100mL, (pcwv) to molarity (M)
    Parameters
    ----------
    pcwv: float, percentage weight/volume, %w/v = % g/mL = g/100mL
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    molarity_M: float, molarity in M
    """
    M = pcwv * 10 / mw
    return M

def mM2pcwv(mM, mw):
    """
    Convert a molarity (mM) to a percentage weight/volume, %w/v = % g/mL = g/100mL
    Parameters
    ----------
    mM: float, molarity in mM
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    pcwv: float, percentage weight/volume, %w/v = % g/mL = g/100mL
    """
    pcwv = mM * mw / 10000
    return pcwv

def M2pcwv(M, mw):
    """
    Convert a molarity (M) to a percentage weight/volume, %w/v = % g/mL = g/100mL
    Parameters
    ----------
    M: float, molarity in M
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    pcwv: float, percentage weight/volume, %w/v = % g/mL = g/100mL
    """
    pcwv = M * mw * 10
    return pcwv

def M2gL(M, mw):
    """
    Convert a molarity (M) to a g/L
    ... mol/L * g/mol = g/L

    Parameters
    ----------
    M: float, molarity in M
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    gL: float, g/L
    """
    gL = M * mw
    return gL
def gL2M(gL, mw):
    """
    Convert a g/L to a molarity (M)
    ... g/L / g/mol = mol/L

    Parameters
    ----------
    gL: float, g/L
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    M: float, molarity in M
    """
    M = gL / mw
    return M

def M2gmL(M, mw):
    """
    Convert a molarity (M) to a g/mL
    Parameters
    ----------
    M: float, molarity in M
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    gmL: float, g/mL
    """
    gmL = M * mw / 1000
    return gmL

def gmL2M(gmL, mw):
    """
    Convert a g/mL to a molarity (M)
    Parameters
    ----------
    gmL: float, g/mL
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    M: float, molarity in M
    """
    M = gmL / mw * 1000
    return M

def mM2gmL(mM, mw):
    """
    Convert a molarity (mM) to a g/mL
    ... (mol/L * 1000) * g/mol = g/L * 1000 = g/mL
    Parameters
    ----------
    mM: float, molarity in mM
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    gmL: float, g/mL
    """
    gmL = mM * mw
    return gmL

def gmL2mM(gmL, mw):
    """
    Convert a molarity (mM) to a g/mL
    ... (mol/L * 1000) * g/mol = g/L * 1000 = g/mL
    Parameters
    ----------
    mM: float, molarity in mM
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    gmL: float, g/mL
    """
    mM = gmL / mw
    return mM

def wv2mM(wv, mw):
    """
    Convert a weight/volume, w/v = g/mL, (wv) to molarity (mM)
    ... Same as gmL2mM
    Parameters
    ----------
    wv: float, weight/volume, w/v = g/mL
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    mM: float, molarity in mM
    """
    mM = gmL2mM(wv, mw)
    return mM

def mM2wv(mM, mw):
    """
    Convert a molarity (mM) to a weight/volume, w/v = g/mL
    ... Same as mM2gmL
    Parameters
    ----------
    mM: float, molarity in mM
    mw: float, molecular weight in Da (g/mol)

    Returns
    -------
    wv: float, weight/volume, w/v = g/mL
    """
    wv = mM2gmL(mM, mw)
    return wv


# HELPERS
def find_nearest(array, value, option='normal'):
    """
    Find an element and its index closest to 'value' in 'array'

    Parameters
    ----------
    array
    value

    Returns
    -------
    idx: index of the array where the closest value to 'value' is stored in 'array'
    array[idx]: value closest to 'value' in 'array'
    """
    if isinstance(array, list):
        array = np.array(array)
    # get the nearest value such that the element in the array is LESS than the specified 'value'
    if option == 'less':
        array_new = copy.copy(array)
        array_new[array_new > value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    # get the nearest value such that the element in the array is GREATER than the specified 'value'
    if option == 'greater':
        array_new = copy.copy(array)
        array_new[array_new < value] = np.nan
        idx = np.nanargmin(np.abs(array_new - value))
        return idx, array_new[idx]
    else:
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]



def sortNArrays(list_of_arrays, element_dtype=tuple):
    """
    Sort a list of N arrays by the order of the first array in the list
    e.g. a=[2,1,3], b=[1,9,8], c=['a', 'b', 'c']
        [a, b, c] -> [(1, 2, 3), (9, 1, 8), ('b', 'a', 'c')]

    Parameters
    ----------
    list_of_arrays: a list of lists/1D-arrays
    element_dtype: data type, default: tuple
        ... This argument specifies the data type of the elements in the returned list
        ... The default data type of the element is tuple because this functon utilizes sorted(zip(...))
        ... E.g. element_dtype=np.ndarray
                -> [a, b, c] -> [np.array([1, 2, 3]),
                                 np.array([9, 1, 8],
                                 np.array(['b', 'a', 'c'], dtype='<U1']

    Returns
    -------
    list_of_sorted_arrays: list of sorted lists/1D arrays

    """

    list_of_sorted_arrays = list(zip(*sorted(zip(*list_of_arrays))))
    if element_dtype == list:
        list_of_sorted_arrays = [list(a) for a in list_of_sorted_arrays]
    elif element_dtype == np.ndarray:
        list_of_sorted_arrays = [np.asarray(a) for a in list_of_sorted_arrays]
    return list_of_sorted_arrays