"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from multicomp.thermodynamics import FloryHuggins


@pytest.mark.parametrize("inert_solvent", [True, False])
def test_flory_huggins(inert_solvent):
    """test FloryHuggins"""
    f = FloryHuggins.from_random_normal(3, 2, 1, inert_solvent=inert_solvent)
    c1, c2 = np.random.uniform(0.0, 0.25, (2, 3))

    if inert_solvent:
        assert len(f.independent_entries) == 3
        assert f.num_entries == 3
    else:
        assert len(f.independent_entries) == 6
        assert f.num_entries == 6

    # calculate thermodynamic quantities
    e0 = f.free_energy_density(c1)
    m0 = f.chemical_potentials(c1)
    p0 = f.pressure(c1)

    # compare to numba compiled version
    calc_vars = f.make_calc_vars()
    e1, m1, p1 = calc_vars(c1)
    _, m2, p2 = calc_vars(c2)
    assert e0 == pytest.approx(e1)
    np.testing.assert_allclose(m0, m1)
    assert p0 == pytest.approx(p1)

    # compare reduced quantities for calculating differences
    calc_diffs = f.make_calc_diffs()
    dm1, dp1 = calc_diffs(c1)
    dm2, dp2 = calc_diffs(c2)
    np.testing.assert_allclose(m1 - m2, dm1 - dm2)
    assert p1 - p2 == pytest.approx(dp1 - dp2)

    assert f.copy().inert_solvent == inert_solvent


@pytest.mark.parametrize("inert_solvent", [True, False])
def test_flory_huggins_modify(inert_solvent):
    """test modifications of the free energy"""
    f1 = FloryHuggins([[0, 1, 2], [1, 0, 3], [2, 3, 0]], inert_solvent=inert_solvent)
    f2 = f1.remove_component(2)
    np.testing.assert_allclose(f2.chis, [[0, 1], [1, 0]])
    f3 = f2.duplicate_component(1)
    np.testing.assert_allclose(f3.chis, [[0, 1, 1], [1, 0, 0], [1, 0, 0]])

    # test reordering of components
    f = FloryHuggins([[0, 1, 2], [1, 0, 2], [2, 2, 0]])
    f.reorder_components()
    np.testing.assert_allclose(f.chis, [[0, 2, 2], [2, 0, 1], [2, 1, 0]])

    f = FloryHuggins.from_random_normal(16, 3, 3)
    ev1 = np.sort(np.linalg.eigvalsh(f.chis))
    f.reorder_components()
    ev2 = np.sort(np.linalg.eigvalsh(f.chis))
    np.testing.assert_allclose(ev1, ev2)


@pytest.mark.parametrize("inert_solvent", [True, False])
def test_setting_values(inert_solvent):
    """test whether modifications are done in-place"""
    f = FloryHuggins.from_random_normal(2, inert_solvent=inert_solvent)
    c1 = f.chis
    c1_values = f.chis.copy()
    f.set_random_chis()
    assert c1 is f.chis and not np.allclose(c1_values, f.chis)

    f = FloryHuggins.from_random_normal(4, inert_solvent=inert_solvent)
    c1 = f.chis
    c1_values = f.chis.copy()
    f.set_block_structure(2, 2, 1)
    assert c1 is f.chis and not np.allclose(c1_values, f.chis)

    assert f.inert_solvent == inert_solvent
    assert f.duplicate_component(0) is not f
    assert f.duplicate_component(0, inplace=True) is f
    assert f.remove_component(0) is not f
    assert f.remove_component(0, inplace=True) is f


@pytest.mark.parametrize("inert_solvent", [True, False])
def test_setting_flat_values(inert_solvent):
    """test whether modifications are done in-place when setting the flat values"""
    f = FloryHuggins.from_random_normal(4, inert_solvent)
    chi_values = f.chis.copy()

    assert len(f.chis_flat) == f.num_entries
    f.chis_flat = f.chis_flat
    np.testing.assert_allclose(f.chis, chi_values)
    f.chis_flat = 0
    np.testing.assert_allclose(f.chis, 0)
    assert len(f.chis_flat) == f.num_entries

    for i in range(f.num_entries):
        f.add_flat(i, 1)
    np.testing.assert_allclose(f.chis, 1.0 - np.eye(f.num_comps))

    f.chis[...] = 0
    for i in range(f.num_entries):
        f.add_flat(i, i)
    np.testing.assert_allclose(f.chis_flat, np.arange(f.num_entries))
