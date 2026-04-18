# TrimCI_Flow — Classical QFlow with TrimCI sub-solver
#
# Coupling levels:
#   uncoupled  : bare fragment integrals, no inter-fragment coupling
#   meanfield  : Fock mean-field coupling via self-consistent γ loop
#   dmet       : full DMET bath embedding (future)
#
# Notebook imports (recommended):
#   from TrimCI_Flow.core import fragment_by_sliding_window, determinant_summary
#   from TrimCI_Flow.uncoupled import run_fragmented_trimci
#   from TrimCI_Flow.meanfield import run_selfconsistent_fragments

# Core shared utilities
from TrimCI_Flow.core.results import FragmentedRunResult
from TrimCI_Flow.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from TrimCI_Flow.core.trimci_adapter import FragmentResult, solve_fragment_trimci, solve_fragment_exact
from TrimCI_Flow.core.analysis import determinant_summary, iteration_summary, convergence_summary

# Coupling-level solvers
from TrimCI_Flow.uncoupled.solver import run_fragmented_trimci
from TrimCI_Flow.meanfield.solver import run_selfconsistent_fragments
from TrimCI_Flow.meanfield.helpers import (
    compute_fragment_rdm1,
    dress_integrals_meanfield,
    assemble_global_rdm1_diag,
)
