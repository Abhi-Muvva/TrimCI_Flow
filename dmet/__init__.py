# TrimCI_Flow/dmet/__init__.py
"""
TrimCI_Flow.dmet -- DMET with TrimCI impurity solver.

Solvers
-------
run_dmet_1shot : 1-shot (non-self-consistent) DMET
run_dmet_sc    : self-consistent DMET (SC-DMET)
"""
from TrimCI_Flow.dmet.solver    import run_dmet_1shot
from TrimCI_Flow.dmet.sc_solver import run_dmet_sc

__all__ = ["run_dmet_1shot", "run_dmet_sc"]
