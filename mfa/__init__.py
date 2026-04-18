# Phase D: MFA-TrimCI — mean-field active TrimCI embedding
from TrimCI_Flow.mfa.solver import run_mfa_d1

try:
    from TrimCI_Flow.mfa.solver import run_mfa_d2
except ImportError:
    run_mfa_d2 = None

__all__ = ["run_mfa_d1", "run_mfa_d2"]
