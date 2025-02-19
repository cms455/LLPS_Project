"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from multicomp.evolution import (
    FullChiIndividual,
    PartitioningEnvironment,
    Population,
    TargetPhaseCountEnvironment,
)


def test_target_phase_count():
    """test whether the TargetPhaseCountEnvironment returns reasonable results"""
    pop = Population([FullChiIndividual(), FullChiIndividual()])
    env = TargetPhaseCountEnvironment()
    env.evolve(pop)


def test_partitioning():
    """test whether the PartitioningEnvironment returns reasonable results"""

    pop = Population([FullChiIndividual(), FullChiIndividual()])
    env = PartitioningEnvironment({"enriched_components": [0]})
    env.evolve(pop)
