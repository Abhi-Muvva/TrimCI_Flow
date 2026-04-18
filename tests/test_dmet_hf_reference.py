# TrimCI_Flow/tests/test_dmet_hf_reference.py
import numpy as np
import pytest


def _tiny_system():
    """2-orbital, 2-electron toy system with known closed-shell HF solution."""
    n_orb, n_elec = 2, 2
    h1 = np.array([[-1.0, -0.1],
                   [-0.1, -0.5]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.6
    eri[1, 1, 1, 1] = 0.4
    eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.3
    eri[0, 1, 0, 1] = eri[1, 0, 1, 0] = 0.1
    eri[0, 1, 1, 0] = eri[1, 0, 0, 1] = 0.1
    return h1, eri, n_elec, n_orb


def test_gamma_mf_shape_and_trace():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    gamma_mf, e_hf = run_hf(h1, eri, n_elec, n_orb)
    assert gamma_mf.shape == (n_orb, n_orb)
    assert np.isclose(np.trace(gamma_mf), n_elec, atol=1e-8), \
        f"Tr(gamma_mf) = {np.trace(gamma_mf):.6f}, expected {n_elec}"


def test_gamma_mf_is_symmetric():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    gamma_mf, _ = run_hf(h1, eri, n_elec, n_orb)
    np.testing.assert_allclose(gamma_mf, gamma_mf.T, atol=1e-10)


def test_e_hf_is_negative_float():
    from TrimCI_Flow.dmet.hf_reference import run_hf
    h1, eri, n_elec, n_orb = _tiny_system()
    _, e_hf = run_hf(h1, eri, n_elec, n_orb)
    assert isinstance(e_hf, float)
    assert e_hf < 0, f"Expected negative RHF energy, got {e_hf}"


def test_no_convergence_warning_on_easy_system():
    """A simple diagonal h1 system should converge without warning."""
    import io, sys
    from TrimCI_Flow.dmet.hf_reference import run_hf
    n_orb, n_elec = 2, 2
    h1 = np.diag([-1.0, 0.5])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 0.4; eri[0, 0, 1, 1] = eri[1, 1, 0, 0] = 0.2; eri[1, 1, 1, 1] = 0.3
    buf = io.StringIO()
    sys.stdout = buf
    run_hf(h1, eri, n_elec, n_orb)
    sys.stdout = sys.__stdout__
    assert "WARNING" not in buf.getvalue(), \
        f"Unexpected convergence warning: {buf.getvalue()}"
