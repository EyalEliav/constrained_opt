import unittest
import numpy as np
from src.utils import plot_lp_results, plot_qp_results
from src.constrained_min import interior_pt
from tests.examples import *


class TestMinimize(unittest.TestCase):
    def test_qp(self):
        ineq_constraints_qp = [qp_ineq1, qp_ineq2, qp_ineq3]
        A = np.array([1, 1, 1]).reshape(1, 3)
        x0 = np.array([0.1, 0.2, 0.7])
        final_candidate, final_obj, history = interior_pt(qp_function, ineq_constraints_qp, A, 0, x0)
        plot_qp_results(history['path'], 'Feasible region and path taken by the algorithm (QP)')

    def test_lp(self):
        ineq_constraints_lp = [lp_ineq1, lp_ineq2, lp_ineq3, lp_ineq4]
        A = None
        x0 = np.array([0.5, 0.75])
        final_candidate, final_obj, history = interior_pt(lp_function, ineq_constraints_lp, A, 0, x0)
        plot_lp_results(history['path'], 'Feasible region and path taken by the algorithm (LP)')