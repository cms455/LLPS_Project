"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from multicomp.mixture import MultiphaseSystem, MultiphaseVolumeSystem
from multicomp.thermodynamics import FloryHuggins


@pytest.mark.parametrize(
    "dist",
    [
        "lognormal",
        "simplex_uniform",
        "simplex_neighborhood",
        "simplex_neighborhood_ring",
    ],
)
def test_setting_values(dist):
    """test whether modifications are done in-place"""
    f = FloryHuggins.from_random_normal(2)
    m = MultiphaseSystem.from_demixed_composition(f)

    f1 = m.free_energy
    c1 = m.phis
    c1_values = m.phis.copy()

    m.set_random_composition(dist=dist)

    assert f1 is m.free_energy
    assert c1 is m.phis and not np.allclose(c1_values, m.phis)

    assert m.num_comps == f.num_comps == 2
    assert m.is_consistent

    f.duplicate_component(0, inplace=True)
    assert m.num_comps == 2 and f.num_comps == 3
    assert not m.is_consistent

    with pytest.raises(AssertionError):
        m.set_random_composition(dist=dist)


def test_angles():
    """test calculating mixing angles"""
    m = MultiphaseSystem(FloryHuggins(np.eye(3)), np.eye(3))
    assert m.num_comps == m.num_phases == 3
    for cluster in [True, False]:
        np.testing.assert_allclose(m.composition_angles(cluster=cluster), np.pi / 2)

    m = MultiphaseSystem(FloryHuggins(np.eye(3)), np.full((3, 3), 0.1))
    assert m.num_comps == m.num_phases == 3
    for cluster in [True, False]:
        np.testing.assert_allclose(m.composition_angles(cluster=cluster), 0, atol=1e-7)


def test_amounts():
    """test dealing with total amounts"""
    f = FloryHuggins.from_random_normal(3, chi_mean=4.0, chi_std=0.5)

    volumes = np.random.uniform(1, 2, 5)
    sys = MultiphaseVolumeSystem.from_random_composition(f, volumes=volumes)
    assert sys.total_amounts.sum() == pytest.approx(sys.total_volume)

    ms = MultiphaseSystem(f, phis=sys.phis)
    sys2 = MultiphaseVolumeSystem.from_amounts(f, ms.phis, sys.total_amounts[:-1])
    np.testing.assert_allclose(sys.phis, sys2.phis)
