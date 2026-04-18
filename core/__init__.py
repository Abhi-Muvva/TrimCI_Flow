# TrimCI_Flow.core — shared infrastructure used by all coupling levels
from TrimCI_Flow.core.fragment import (
    fragment_by_sliding_window,
    extract_fragment_integrals,
    fragment_electron_count,
)
from TrimCI_Flow.core.trimci_adapter import (
    FragmentResult,
    solve_fragment_trimci,
    solve_fragment_exact,
)
from TrimCI_Flow.core.results import FragmentedRunResult
from TrimCI_Flow.core.analysis import (
    determinant_summary,
    iteration_summary,
    convergence_summary,
)
