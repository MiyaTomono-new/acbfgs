"""
acbfgs — Adaptive Cautious BFGS Methods for Nonconvex Optimization.
==================================================================

Algorithm, implementation and reproducible benchmarking.
"""

__version__ = "1.0.0"
__author__ = "Author Name"

from .algorithms import (
    acbfgs, bfgs_standard, cbfgs_lf, damped_bfgs_powell,
    lbfgs, gradient_descent, SOLVERS, SOLVERS_ABLATION,
)
from .test_problems import TestFunction, build_problem_suite
from .test_problems_extended import build_extended_suite
from .test_problems_batch3 import build_full_suite
