"""
This script show a single instance of calculating phase count for random interaction
matrices, using the field-like dynamics.

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import time

import numpy as np

import multicomp as mm

num_comp: int = 8  # number of components
chi_mean: float = 0.0  # mean value of the interaction
chi_std: float = 10.0  # standard deviation of the interaction
comp_dist: str = "simplex_uniform"  # distribution of the inital composition
comp_sigma: float = 0.0  # breadth of the random composition distribution
num_phases: int = 64  # number of initial phases to try
t_range: float = 1e6
seed_chi = None  # seed of the P
seed_mix = None  # seed of the random number generator for choosing mixtures

if num_phases is None:
    num_phases = num_comp + 2
evolve_params = {
    "t_range": t_range,
    "dt": 1,
    "interval": 10000.0,
    "tolerance": 1e-5,
    "progress": True,
    "save_intermediate_data": False,
}

# iterate over different realizations of the interaction matrix chis
diagnostics = {"reached_final_time": 0, "simulation_aborted": 0, "t_final": []}
start = time.time()

# choose a random free energy
rng_chi = None if seed_chi is None else np.random.default_rng(seed_chi)
f = mm.FloryHuggins.from_random_normal(num_comp, chi_mean, chi_std, rng=rng_chi)

# prepare simulation
rng_mix = None if seed_mix is None else np.random.default_rng(seed_mix)
m = mm.MultiphaseVolumeSystem.from_random_composition(
    f, np.ones(num_phases), dist=comp_dist, sigma=comp_sigma, rng=rng_mix
)

r = mm.FieldLikeRelaxationDynamics(m)

# choose a new initial composition
r.mixture.free_energy.set_random_chis(chi_mean, chi_std, rng=rng_chi)
r.mixture.set_random_composition(dist=comp_dist, sigma=comp_sigma, rng=rng_mix)
# use same initial composition for all phases, guarantee that total fractions are random.
# FieldLikeRelaxationDynamics will randomize phases on its own way
for itr_phase in range(1, num_phases):
    r.mixture.phis[itr_phase] = r.mixture.phis[0]


# run simulation
try:
    ts, result = r.evolve(**evolve_params)
except RuntimeError:
    # simulation could not finish
    diagnostics["simulation_aborted"] += 1
else:
    # finalize
    if np.isclose(ts, t_range):
        diagnostics["reached_final_time"] += 1
    diagnostics["t_final"].append(ts)

# add diagnostic information
t_final = np.array(diagnostics["t_final"])
diagnostics["t_final"] = {
    "min": float(t_final.min()),
    "mean": float(t_final.mean()),
    "std": float(t_final.std()),
    "max": float(t_final.max()),
}
diagnostics["runtime"] = time.time() - start

print(result.count_clusters(), "clusters found")
print(result.entropy_production)
