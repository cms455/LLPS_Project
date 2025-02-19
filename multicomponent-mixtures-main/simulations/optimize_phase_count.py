#!/usr/bin/env python3 -m modelrunner
"""
This script optimizes interaction matrices with respect to the phase count.

Run `./optimize_phase_count.py -h` to get information about the permissible parameters. 

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import os.path
import sys

sys.path.append(os.path.expanduser("~/Code/multicomponent-mixtures"))
sys.path.append(os.path.expanduser("~/Code/py-modelrunner"))

from modelrunner import submit_job

from multicomp import (
    FullChiIndividual,
    TargetPhaseCountEnvironment,
    get_optimization_model,
)

# define the evolution model by composing classes for individual and environment
EvolutionModel = get_optimization_model(FullChiIndividual, TargetPhaseCountEnvironment)


if __name__ == "__main__":
    # submit this file as a modelrunner, which will detect the EvolutionModel automatically
    submit_job(__file__, output="result.hdf5", method="qsub")
