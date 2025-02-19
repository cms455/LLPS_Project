"""
Package containing classes to run evolutionary experiments

.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from .environment import PartitioningEnvironment, TargetPhaseCountEnvironment
from .individual import FullChiIndividual
from .optimization import get_optimization_model
from .population import Population
