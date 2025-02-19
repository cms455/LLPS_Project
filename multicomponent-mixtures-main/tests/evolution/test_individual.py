"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np

from multicomp.evolution.individual import FullChiIndividual


def test_cross():
    """test crossover of individuals"""
    i1 = FullChiIndividual()
    i2 = i1.cross(i1)
    assert i1 is not i2
    np.testing.assert_array_equal(i1.free_energy.chis, i2.free_energy.chis)


def test_reproduction():
    """test whether reproduction copies structures correctly"""
    i1 = FullChiIndividual({"repetitions": 0})
    i2 = i1.copy()

    assert i1 is not i2
    assert i1.free_energy is not i2.free_energy
    assert np.array_equal(i1.free_energy.chis, i2.free_energy.chis)

    i1.get_ensemble([])
    i2.get_ensemble([])

    assert i1.dynamics is not i2.dynamics
    assert i1.dynamics.mixture is not i2.dynamics.mixture
    assert i1.dynamics.mixture.free_energy is not i2.dynamics.mixture.free_energy


def test_mutation():
    """test whether mutation leaves structures intact"""
    i1 = FullChiIndividual()

    i1_c = i1.free_energy.chis.copy()
    i1_f = i1.free_energy
    i1_d = i1.dynamics

    i1.mutate()

    assert i1_f is i1.free_energy
    assert not np.allclose(i1_c, i1.free_energy.chis)
    assert i1_d is i1.dynamics


def test_individual():
    """test whether an individual returns a result"""
    i1 = FullChiIndividual()
    assert sum(i1.get_ensemble("phase_counts", progress=False)["phase_counts"]) > 0
