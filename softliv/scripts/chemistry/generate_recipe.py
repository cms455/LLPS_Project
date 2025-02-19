"""
Author: Teagan and Takumi
Last Updated: 2023/11/27
Description: This script generates a recipe for a given sample based on the stock concentrations, final concentrations, and dilutions.
How to use:
    1. Edit the recipe files in the recipe directory.
        - /.../recipe/stockConcentrations.csv: Stock concentrations of each ingredient
        - /.../recipe/finalConcentrations.csv: Final concentrations of each ingredient
        - /.../recipe/dilutions.csv: Dilution factors of each ingredient

    2. Run a script with the following command line arguments.

        python generate_recipe.py -i <input directory> -o <output directory> -v <total volume> -header <header>

            -i: Input directory where recipe files are stored
            -o: Output directory for saving the recipe files. Default is the desktop.
            -v: Total volume of a sample in µL
            -header: Header for the recipe file. Default is 'BSA_PEG'
"""

import numpy as np
import pandas as pd
from datetime import datetime
import os
from pathlib import Path
import argparse
import ast
#
script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)

def read_recipe(recipeDir):
    def format(d):
        d['Concentration'] = d['Concentration'].apply(lambda x: ast.literal_eval(f"[{x}]"))
        d['Concentration'] = d['Concentration'].apply(lambda x: x[0] if len(x) == 1 else x)
        return d

    fnStockConc = Path(recipeDir) / 'stockConcentrations.csv'
    fnFinalConc = Path(recipeDir) / 'finalConcentrations.csv'
    fnDilutions = Path(recipeDir) / 'dilutions.csv'

    dfStockConc = format(pd.read_csv(fnStockConc))
    dfFinalConc = format(pd.read_csv(fnFinalConc))
    dfDilutions = format(pd.read_csv(fnDilutions))

    dictStockConc = {row['Name']: row['Concentration'] for index, row in dfStockConc.iterrows()}
    dictFinalConc = {row['Name']: row['Concentration'] for index, row in dfFinalConc.iterrows()}
    dictDilutions = {row['Name']: row['Concentration'] for index, row in dfDilutions.iterrows()}

    return dictStockConc, dictFinalConc, dictDilutions


def calculate_recipe(stockConcentrations, finalConcentrations, dilutions, sampleVolumes):
    """
    Calculate the recipe for a given sample based on the stock concentrations, final concentrations, and dilutions.

    Parameters
    ----------
    stockConcentrations: dict, stock concentrations of each ingredient
    finalConcentrations: dict, final concentrations of each ingredient
    dilutions: dict, dilution factors of each ingredient
    sampleVolumes: float, total volume of a sample in µL

    Returns
    -------
    recipe: pd.DataFrame, recipe for a given sample
    """
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
        for i in range(len(finalConcentrations[variableParameters[0]])):
            # Create labels for the row based on variable parameters
            labels = [str(round(finalConcentrations[value][i], 3)) + ' ' + str(value) for value in variableParameters]
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
                # non variable parameter
                else:
                    subRecipe[k] = sampleVolumes * finalConcentrations[ingredient] / (
                                stockConcentrations[ingredient] / dilutions[ingredient])
                    volume += subRecipe[k]

            # Calculate the buffer volume needed to reach the correct total volume
            buffer = sampleVolumes - volume
            # Add the sub recipe to the list of recipes
            recipe.loc[label] = subRecipe + [buffer] + [sampleVolumes]

    else:
        label = 'µL'
        # A sub recipe is a single recipe
        subRecipe = [0] * len(ingredientsAndDilutions)
        volume = 0
        for k, ingredient in enumerate(ingredientsAndDilutions):
            if 'dilution' in ingredient:
                subRecipe[k] = dilutions[ingredient.replace(' dilution', '')]
            else:
                subRecipe[k] = sampleVolumes * finalConcentrations[ingredient] / (
                            stockConcentrations[ingredient] / dilutions[ingredient])
                volume += subRecipe[k]
        # Calculate the buffer volume needed to reach the correct total volume
        buffer = sampleVolumes - volume
        # Add the sub recipe to the list of recipes
        recipe.loc[label] = subRecipe + [buffer] + [sampleVolumes]

    # If any dilution rows are 1x for all recipes, don't show it
    condition = [column for column in recipe.columns if ' dilution' in column and all(recipe[column] == 1)]
    recipe = recipe.drop(columns=condition)
    # Show ingredients as rows, each recipe as a column
    recipe = recipe.transpose().round(decimals=2)
    return recipe
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
    parser.add_argument('-i', dest='recipeDir', type=str,
                        default=Path(script_dir) / 'recipe',
                        help='Input directory where recipe files are stored')
    parser.add_argument('-o', dest='outputDir', type=str,
                        default=Path(os.path.expanduser("~/Desktop")),
                        help='Output directory for saving the recipe files')
    parser.add_argument('-v', dest='volume', type=float, default=1000.,
                        help='Total volume of a sample in µL')
    parser.add_argument('-header', dest='header', type=str, default='BSA_PEG',
                        help='Header for the recipe file')

    args = parser.parse_args()
    return args

def isVariable(variable):
    """
    Check if a variable is a list or numpy array

    Parameters
    ----------
    variable

    Returns
    -------
    bool, True if variable is a list or numpy array
    """
    return isinstance(variable,np.ndarray) or isinstance(variable,list)


def main(args):
    """
    Main function

    Returns
    -------
    None
    """
    stockConcentrations, finalConcentrations, dilutions = read_recipe(args.recipeDir)
    recipe = calculate_recipe(stockConcentrations, finalConcentrations, dilutions, args.volume)
    outputFile = Path(args.outputDir) / (datetime.today().strftime('%Y-%m-%d') + f'_{os.path.split(args.recipeDir)[1]}.csv')

    if not os.path.exists(args.outputDir):
        os.makedirs(args.outputDir)
    recipe.to_csv(outputFile)
    print("Recipe saved to: " + outputFile.__str__() + "\n")
    print(recipe)

if __name__ == "__main__":
    args = parse_args()
    main(args)