import numpy as np
import sys
sys.path.append('/Users/calvinsmith/dufresne_lab/multicomponent-mixtures-main')
from datetime import datetime
import flory
import h5py
import itertools
import matplotlib.pyplot as plt

def generate_combinations_2(num_components, step_size):
    
    combinations = []

    
    def generate(current, remaining, depth):
        if depth == num_components - 1:
            combinations.append(current + [remaining])
            return
        for i in np.arange(0, remaining + step_size, step_size):
            generate(current + [i], remaining - i, depth + 1)

    generate([], 1.0, 0)
    return np.array(combinations)


def replace_negative(matrix,small_val):
    # Replace negative values in the matrix with 0.01
    modified_matrix = np.where(matrix < 0, small_val, matrix)
    return modified_matrix

def thresh(arr, threshold):
    # Create a boolean mask where True indicates that all elements in the row are above the threshold
    mask = np.all(arr >= threshold, axis=1)

    # Use the mask to select rows that meet the condition
    return arr[mask]

def thresh_2(arr, threshold):
    # Initialize an empty list to hold rows that meet the condition
    filtered_rows = []

    # Loop through each row in the array
    for row in arr:
        # Assume all elements are above the threshold initially
        keep_row = True
        
        # Check each element in the row
        for element in row:
            if element < threshold:
                keep_row = False
                break  # Exit the loop early if any element fails the condition

        # If all elements are above the threshold, add the row to the filtered list
        if keep_row:
            filtered_rows.append(row)

    # Return the list of filtered rows as a NumPy array
    return np.array(filtered_rows)


def random_interaction_matrix(
    num_comp: int, chi_mean: float = None, chi_std: float = 1
) -> np.ndarray:
    """create a random interaction matrix

    Args:
        num_comp (int): The component count
        chi_mean (float): The mean interaction strength
        chi_std (float): The standard deviation of the interactions

    Returns:
        The full, symmetric interaction matrix
    """
    if chi_mean is None:
        chi_mean = 3 + 0.4*num_comp
        #chi_mean = 10+0.4*num_comp
        #chi_mean = 0.0001

    # initialize interaction matrix
    chis = np.zeros((num_comp, num_comp))

    # determine random entries
    num_entries = num_comp * (num_comp - 1) // 2
    chi_vals = np.random.normal(chi_mean, chi_std, num_entries)

    # build symmetric  matrix from this
    i, j = np.triu_indices(num_comp, 1)
    chis[i, j] = chi_vals
    chis[j, i] = chi_vals
    
   # chis = np.array([[0,4,4],[4,0,4],[4,4,0]])
    
    return chis

def most_dilute_flory(phase):
    dilute_phase_vol_index = np.argmax(phase.volumes)
    return phase.fractions[dilute_phase_vol_index]



def chi_matrix_w_sol(num_comps,chi_strength):
    num_comps = num_comps -1
    chi_matrix = random_interaction_matrix(num_comps,chi_strength)
    chi_matrix = np.vstack([chi_matrix,np.zeros(num_comps)])
    zero_col = np.zeros(num_comps+1)
    zero_col = zero_col[:, np.newaxis]
    chi_matrix = np.hstack([chi_matrix,zero_col])
    return chi_matrix

def chi_matrix_strong_out(num_comps,num_out,chi_strength,out_chi_strength):
    num_comps = num_comps -1
    chi_matrix = random_interaction_matrix(num_comps,chi_strength)
    out_chi_matrix = random_interaction_matrix(num_comps,out_chi_strength)
    chi_matrix[:,num_comps - num_out:num_comps] = out_chi_matrix[:,num_comps - num_out:num_comps]
    chi_matrix[num_comps - num_out:num_comps:] = out_chi_matrix[num_comps - num_out:num_comps:]
    chi_matrix = np.vstack([chi_matrix,np.zeros(num_comps)])
    zero_col = np.zeros(num_comps+1)
    zero_col = zero_col[:, np.newaxis]
    chi_matrix = np.hstack([chi_matrix,zero_col])
    return chi_matrix
    
# This function tells it whether it is in bounds so we don't always have to add constraints
#To the optimization function, makes it easier
def in_bounds(phi_H, phi_H_thresh):
    sum = 0
    for phi in phi_H:
        sum += phi
        if phi < 0:
            return False
    if sum > phi_H_thresh:
        return False
    else:
        return True
    
def get_num_of_phases(phi_vector,fh_energy):
    dynamics_params = {
    "initialization_random_std": 5,  # how random the initial guess is
    "acceptance_Js": 0.0002, # how fast the relative volumes are evolved
    "acceptance_omega": 0.002, # how fast the compositions are evolved
    }

    evolve_params = {
    "t_range": 1000000,  # maximum iteration steps
    "dt": 1,  # always unity for FieldLikeRelaxationDynamics
    "interval": 10000,  # the frequency for checking convergence
    "tolerance": 1e-5,  # criteria for determining convergence
    "progress": False,
    "save_intermediate_data": False,
    }
    num_compartments = 64
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy, 
        np.array(phis), # compositions of each compartment
        np.ones(num_compartments) / num_compartments # relative volume of each compartment
    )    
    
    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)
    
    

def get_num_of_phases_fast(phi_vector,fh_energy):
    dynamics_params = {
    "initialization_random_std": 5,  # how random the initial guess is
    "acceptance_Js": 0.0002, # how fast the relative volumes are evolved
    "acceptance_omega": 0.002, # how fast the compositions are evolved
    }

    evolve_params = {
    "t_range": 100000,  # maximum iteration steps
    "dt": 1,  # always unity for FieldLikeRelaxationDynamics
    "interval": 10000,  # the frequency for checking convergence
    "tolerance": 1e-5,  # criteria for determining convergence
    "progress": False,
    "save_intermediate_data": False,
    }
    num_compartments = 64
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy, 
        np.array(phis), # compositions of each compartment
        np.ones(num_compartments) / num_compartments # relative volume of each compartment
    )    
    
    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)
    
    
    

def get_uniform_random_composition(num_phases: int, num_comps: int) -> np.ndarray:
    """pick concentrations uniform from allowed simplex (sum of fractions < 1)

    Args:
        num_phases (int): the number of phases to pick concentrations for
        num_comps (int): the number of components to use

    Returns:
        The fractions of num_comps components in num_phases phases
    """
    phis = np.empty((num_phases, num_comps))
    for n in range(num_phases):
        phi_max = 1.0
        for d in range(num_comps):
            x = np.random.beta(1, num_comps - d) * phi_max
            phi_max -= x
            phis[n, d] = x
    return phis


def square_matrix(n, x):
    # Create an nxn matrix of x's
    matrix = np.full((n, n), x)

    # Set the diagonal elements to 0
    np.fill_diagonal(matrix, 0)

    return matrix


def get_num_of_phases_v2(phi_vector, fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return len(unique_phases)

 
    
def evolve_phases(phi_vector,fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    unique_phases = result.get_clusters()
    return unique_phases

def evolve_phases_result(phi_vector,fh_energy, compartments):
    dynamics_params = {
        "initialization_random_std": 5,  # how random the initial guess is
        "acceptance_Js": 0.0002,  # how fast the relative volumes are evolved
        "acceptance_omega": 0.002,  # how fast the compositions are evolved
    }

    evolve_params = {
        "t_range": 1000000,  # maximum iteration steps
        "dt": 1,  # always unity for FieldLikeRelaxationDynamics
        "interval": 10000,  # the frequency for checking convergence
        "tolerance": 1e-5,  # criteria for determining convergence
        "progress": False,
        "save_intermediate_data": False,
    }
    num_compartments = compartments
    phis = [phi_vector] * num_compartments
    mixture = mm.MultiphaseVolumeSystem(
        fh_energy,
        np.array(phis),  # compositions of each compartment
        np.ones(num_compartments) / num_compartments  # relative volume of each compartment
    )

    dynamics = mm.FieldLikeRelaxationDynamics(mixture, parameters=dynamics_params)
    ts, result = dynamics.evolve(**evolve_params)
    return result


def timing_chi(comp_range,chi):
    num_comps = 3
    time_data = np.zeros((comp_range,2))
    
    for i in range(comp_range):
    
        print(num_comps)
        chi_matrix = square_matrix(num_comps,chi)
        start_time = datetime.now()  # Record start time

        time_data[i][0] = num_comps
        from tqdm.notebook import tqdm
        print("start: " + str(num_comps))
        print("Init ternary_matrix")
        norm_array = normal_array(5,num_comps)
        for row in tqdm(norm_array):

            # Apply the arbitrary function to the row and add the result to the second column
            fh_energy = mm.FloryHuggins(np.array(chi_matrix))
            test = get_num_of_phases_fast(row,fh_energy)
            #print(ternary_matrix[j,1])

        end_time = datetime.now()  # Record end time
        duration = end_time - start_time  # Calculate duration
        total_seconds = duration.total_seconds()
        time_data[i][1] = total_seconds  # Store total seconds instead of string
        print(duration)
        num_comps += 1
        
    return time_data



def timing_rand_chi(comp_range,avg_chi):
    num_comps = 3
    time_data = np.zeros((comp_range,2))
    
    for i in range(comp_range):
    
        print(num_comps)
        chi_matrix = random_interaction_matrix(num_comps,avg_chi)
        start_time = datetime.now()  # Record start time

        time_data[i][0] = num_comps
        from tqdm.notebook import tqdm
        print("start: " + str(num_comps))
        print("Init ternary_matrix")
        norm_array = normal_array(5,num_comps)
        for row in tqdm(norm_array):

            # Apply the arbitrary function to the row and add the result to the second column
            fh_energy = mm.FloryHuggins(np.array(chi_matrix))
            test = get_num_of_phases_fast(row,fh_energy)
            #print(ternary_matrix[j,1])

        end_time = datetime.now()  # Record end time
        duration = end_time - start_time  # Calculate duration
        total_seconds = duration.total_seconds()
        time_data[i][1] = total_seconds  # Store total seconds instead of string
        print(duration)
        num_comps += 1
        
    return time_data


def normal_array(num_rows, num_comps):
    random_matrix = np.random.rand(num_rows, num_comps)
    
    # Normalize each row to sum to 1
    row_sums = random_matrix.sum(axis=1)[:, np.newaxis]  
    normalized_matrix = random_matrix / row_sums
    return normalized_matrix

def generate_simplex_grid(num_comps, num_points):
    """Generates a properly spaced grid over the num_comps-simplex."""
    
    # Choose an integer N for the lattice grid spacing
    N = int(round(num_points ** (1 / (num_comps - 1))))  # Estimate good spacing

    # Generate integer partitions of N into num_comps parts
    partitions = itertools.combinations_with_replacement(range(N + 1), num_comps - 1)

    grid = []
    for partition in partitions:
        full_partition = [partition[0]] + [partition[i] - partition[i-1] for i in range(1, len(partition))] + [N - partition[-1]]
        grid.append(np.array(full_partition) / N)  # Normalize to sum to 1

    grid = np.array(grid)
    
    # Shuffle and select the required number of points
    np.random.shuffle(grid)
    return grid[:num_points]  # Return the requested number of points



def peak_file(file_path):
    with h5py.File(file_path, 'r') as hf:
        for chi_key in hf.keys():
            print(f"\n📂 {chi_key}:")  # Print the chi_matrix group name
            
            g1 = hf[chi_key]
            
            # Print initial data
            print("  ├── initial_points:", g1["initial_points"][:].shape)
            print("  ├── chi_matrix:", g1["chi_matrix"][:].shape)
            
            # Access evolved phases
            if "evolved_phases" in g1:
                g2 = g1["evolved_phases"]
                print("  ├── evolved_phases:")
                print("      ├── volumes:", g2["volumes"][:].shape)
                print("      ├── comp_fracs:", g2["comp_fracs"][:].shape)
                print("      ├── num_phases:", g2["num_phases"][:].shape)
                
                # Optionally print a small sample of the data
                print("      ├── num_phases sample:", g2["num_phases"][:5])  # First 5 values

                
def analyze_phase_behavior(folder_path, chi_function, phase_metric_function, plot_title, x_label, y_label):
    """
    Generalized function to analyze phase behavior based on arbitrary functions applied to the chi matrix and phase space.

    Parameters:
    - folder_path (str): Path to the directory containing HDF5 files.
    - chi_function (function): Function that extracts a single numeric value from the chi matrix.
    - phase_metric_function (function): Function that computes a single numeric metric from the phase space data.
    - plot_title (str): Title of the generated plot.
    - x_label (str): Label for the x-axis (describes the chi metric).
    - y_label (str): Label for the y-axis (describes the phase metric).
    """
    
    chi_values = []
    phase_metrics = []

    for file in os.listdir(folder_path):
        if file.endswith(".h5"):
            file_name = os.path.join(folder_path, file)
            try:
                with h5py.File(file_name, 'r') as hf:
                    for chi_key in hf.keys():  # Iterate through chi_matrix groups
                        g1 = hf[chi_key]

                        if "evolved_phases" in g1:
                            g2 = g1["evolved_phases"]
                            chi_matrix = g1["chi_matrix"][:]
                            phase_data = {
                                "volumes": g2["volumes"][:],
                                "num_phases": g2["num_phases"][:],
                            }

                            # Apply user-defined functions
                            chi_value = chi_function(chi_matrix)
                            phase_metric = phase_metric_function(phase_data)

                            # Store values
                            chi_values.append(chi_value)
                            phase_metrics.append(phase_metric)
            except OSError as e:
                print(f"❌ Corrupted file: {file} - {e}")

    # Convert lists to numpy arrays for plotting
    chi_values = np.array(chi_values)
    phase_metrics = np.array(phase_metrics)

    # Sort data for better visualization
    sorted_indices = np.argsort(chi_values)
    chi_values = chi_values[sorted_indices]
    phase_metrics = phase_metrics[sorted_indices]

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.scatter(chi_values, phase_metrics, color='blue', alpha=0.75, label="Data points")
    plt.plot(chi_values, phase_metrics, linestyle='dashed', color='red', alpha=0.5, label="Trend")

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(plot_title)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


def analyze_volume_fractions(folder_path, chi_function, plot_title, x_label):
    """
    Generalized function to analyze phase behavior by sorting based on an arbitrary function applied to the chi matrix
    and outputting a stacked bar chart of phase volume fractions.

    Parameters:
    - folder_path (str): Path to the directory containing HDF5 files.
    - chi_function (function): Function that extracts a single numeric value from the chi matrix for sorting.
    - plot_title (str): Title of the generated plot.
    - x_label (str): Label for the x-axis (describes the chi metric).
    """

    chi_values = []
    volume_fractions = []

    for file in os.listdir(folder_path):
        if file.endswith(".h5"):
            file_name = os.path.join(folder_path, file)
            try:
                with h5py.File(file_name, 'r') as hf:
                    chi_values_per_file = []
                    volume_fractions_per_file = []

                    for chi_key in hf.keys():  # Iterate through chi_matrix groups
                        g1 = hf[chi_key]

                        if "evolved_phases" in g1:
                            g2 = g1["evolved_phases"]
                            chi_matrix = g1["chi_matrix"][:]
                            phase_data = {
                                "volumes": g2["volumes"][:],
                            }

                            # Compute sorting metric and volume fractions
                            chi_value = chi_function(chi_matrix)
                            volumes = phase_data["volumes"]

                            # Normalize each volume set to sum to 1
                            total_volume = np.sum(volumes, axis=-1, keepdims=True)
                            normalized_volumes = volumes / total_volume
                            avg_fractions = np.mean(normalized_volumes, axis=0)  # Compute mean across all points

                            # Store values for this file
                            chi_values_per_file.append(chi_value)
                            volume_fractions_per_file.append(avg_fractions)

                    # Compute the average for the entire file
                    if chi_values_per_file and volume_fractions_per_file:
                        chi_values.append(np.mean(chi_values_per_file))
                        volume_fractions.append(np.mean(volume_fractions_per_file, axis=0))

            except OSError as e:
                print(f"❌ Corrupted file: {file} - {e}")

    # Convert lists to numpy arrays for plotting
    chi_values = np.array(chi_values)
    volume_fractions = np.array(volume_fractions)

    # Sort data based on chi_function
    sorted_indices = np.argsort(chi_values)
    chi_values = chi_values[sorted_indices]
    volume_fractions = volume_fractions[sorted_indices]

    # Plot stacked bar chart
    num_phases = volume_fractions.shape[1]
    ind = np.arange(len(chi_values))  # X-axis positions

    plt.figure(figsize=(10, 6))
    bottom = np.zeros(len(chi_values))  # Initialize bottom for stacking

    for i in range(num_phases):
        plt.bar(ind, volume_fractions[:, i], bottom=bottom, label=f"Phase {i+1}")
        bottom += volume_fractions[:, i]

    plt.xticks(ind, [f"{val:.2f}" for val in chi_values], rotation=45, ha="right")
    plt.xlabel(x_label)
    plt.ylabel("Volume Fraction (Stacked)")
    plt.title(plot_title)
    plt.legend(title="Phases")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.show()
