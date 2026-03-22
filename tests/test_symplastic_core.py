"""Smoke tests for the symplastic_growth package (Track A — reproducible installs)."""

import numpy as np

from symplastic_growth import GrowthParams, posm, pturg, SymplasticSimulator


def test_posm_pturg_finite():
    p = GrowthParams(alph=10.0)
    assert posm(1.2, 1.0, p.alph) > 0
    t = pturg(1.1, 1.0, 19.0, p.m_young, p.s_cw)
    assert np.isfinite(t)


def test_growth_params_defaults():
    p = GrowthParams()
    assert p.n_cell_files >= 1
    assert len(p.n_cells_per_file) == p.n_cell_files


def test_simulator_few_steps():
    """Minimal integration: a few steps complete without raising (keeps CI fast)."""
    params = GrowthParams(alph=10.0, etha=0.15, thresh=2.0, n_cell_files=1)
    params.n_cells_per_file = [4]
    params.cell_width = [19.0]
    sim = SymplasticSimulator(params, n_cells=4, dt=1.0)
    res = sim.run_until_length(Ly=1e9, max_steps=5)  # stop early via max_steps
    assert len(res.t) >= 2
    assert res.total_length[-1] > 0


def test_surrogate_import():
    from surrogate import MLPSurrogate, generate_training_data

    assert MLPSurrogate is not None
    assert generate_training_data is not None
