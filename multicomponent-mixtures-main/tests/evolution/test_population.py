"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

from multicomp.evolution import FullChiIndividual, Population


def test_population():
    """test whether the population returns reasonable results"""
    pop = Population([FullChiIndividual(), FullChiIndividual()])
    stats = pop.get_stats("phase_counts")
    assert sum(sum(stat["phase_counts"]) for stat in stats) > 0
    pop.interaction_matrix_stats()
