"""
.. codeauthor:: David Zwicker <david.zwicker@ds.mpg.de>
"""

import numpy as np
import pytest

from multicomp.dynamics import RelaxationDynamics
from multicomp.mixture import MultiphaseSystem
from multicomp.thermodynamics import FloryHuggins


def test_setting_values():
    """test whether modifications are done in-place"""
    f = FloryHuggins.from_random_normal(2)
    m = MultiphaseSystem.from_demixed_composition(f)
    r = RelaxationDynamics(m)

    f1 = m.free_energy
    c1_values = m.free_energy.chis.copy()
    p1 = m.phis
    p1_values = m.phis.copy()

    t, m2 = r.evolve(t_range=10.0, dt=1e-3, progress=False)
    assert t > 0
    assert p1 is r.mixture.phis and np.allclose(p1_values, r.mixture.phis)
    assert p1 is not m2.phis and not np.allclose(p1_values, m2.phis)

    f.set_random_chis()
    assert not np.allclose(c1_values, r.mixture.free_energy.chis)
    assert f1 is r.mixture.free_energy

    m.set_random_composition(dist="lognormal")
    assert p1 is r.mixture.phis and not np.allclose(p1_values, r.mixture.phis)


@pytest.mark.parametrize("method", ["explicit", "implicit"])
def test_convergence(method):
    """test whether the relaxation actually converges to a stationary state"""
    f = FloryHuggins.from_random_normal(5)
    m = MultiphaseSystem.from_demixed_composition(f)
    r = RelaxationDynamics(m, {"method": method})
    _, res = r.evolve(t_range=1e3, dt=1e-3, tolerance=1e-16, progress=False)

    np.testing.assert_allclose(res.pressures, res.pressures.mean(), rtol=1e-4)
    mus = res.chemical_potentials
    mu_mean = mus.mean(axis=0)
    for mu in mus:
        np.testing.assert_allclose(mu, mu_mean, rtol=1e-4, atol=1e-4)


def test_solver_methods():
    """test whether implicit and explicit simulations give the same result"""
    f = FloryHuggins.from_random_normal(2)
    m = MultiphaseSystem.from_demixed_composition(f)
    r_e = RelaxationDynamics(m, {"method": "explicit"})
    r_ea = RelaxationDynamics(m, {"method": "explicit_adaptive"})
    r_i = RelaxationDynamics(m, {"method": "implicit"})

    p1 = m.phis
    p1_values = m.phis.copy()

    t_e, m_e = r_e.evolve(t_range=10.0, dt=1e-3, progress=False)
    _, m_ea = r_ea.evolve(t_range=10.0, dt=1e-3, progress=False)
    t_i, m_i = r_i.evolve(t_range=10.0, dt=1e-1, progress=False)
    assert t_e == pytest.approx(t_i, abs=1.5)
    assert p1 is r_e.mixture.phis and np.allclose(p1_values, r_e.mixture.phis)
    assert p1 is r_ea.mixture.phis and np.allclose(p1_values, r_ea.mixture.phis)
    assert p1 is r_i.mixture.phis and np.allclose(p1_values, r_i.mixture.phis)
    np.testing.assert_allclose(m_e.phis, m_ea.phis, rtol=5e-2, atol=5e-2)
    np.testing.assert_allclose(m_e.phis, m_i.phis, rtol=5e-2, atol=5e-2)


def test_convergence_criteria():
    """test whether convergence criteria are all viable"""
    f = FloryHuggins.from_random_normal(2)
    m = MultiphaseSystem.from_demixed_composition(f)
    r_1 = RelaxationDynamics(m, {"convergence_criterion": "composition_change"})
    r_2 = RelaxationDynamics(m, {"convergence_criterion": "entropy_production"})

    p1 = m.phis
    p1_values = m.phis.copy()

    _, m_1 = r_1.evolve(t_range=10.0, dt=1e-3, tolerance=1e-10, progress=False)
    _, m_2 = r_2.evolve(t_range=10.0, dt=1e-3, tolerance=1e-10, progress=False)
    assert p1 is r_1.mixture.phis and np.allclose(p1_values, r_1.mixture.phis)
    assert p1 is r_2.mixture.phis and np.allclose(p1_values, r_2.mixture.phis)
    np.testing.assert_allclose(m_1.phis, m_2.phis, rtol=1e-5, atol=1e-5)
