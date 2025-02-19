"""
Package defining multiphase systems and their relaxation dynamics

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .dynamics import RelaxationDynamics, plot_concentrations, plot_convergence
from .dynamics_field_like import FieldLikeRelaxationDynamics
from .evolution import *
from .mixture import Mixture, MultiphaseSystem, MultiphaseVolumeSystem
from .mixture import get_uniform_random_composition_np as get_uniform_random_composition
from .thermodynamics import FloryHuggins

__version__ = "0.3"
