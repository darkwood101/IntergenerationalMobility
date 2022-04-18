import unittest

from aamodel.solver import mdp_solver, uniform_dist
import matplotlib.pyplot as plt
import numpy as np


class solver_test(unittest.TestCase):
    def test_R(self):
        """
        u = uniform_dist()
        s = mdp_solver(dist = u,
                       sigma = 0.4,
                       tau = 0.05,
                       p_A = 0.03,
                       p_D = 0.02,
                       N = 1000,
                       gamma = 0.99,
                       alpha = 0.05,
                       epsilon = 0.0001)
        plt.plot(s.R[-1, :])
        plt.show()
        Q = s.run()
        # Q[state, policy]
        print(Q[700])
        plt.plot(Q.argmax(axis=1)[1:-1])
        plt.show()
        """

        u = uniform_dist()
        lower_bounds = []
        upper_bounds = []
        for n_unpriv in range(1000):
            lower, upper = u.action_bounds(n_unpriv / 1000, 0.4, 0.15)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
        plt.plot(lower_bounds)
        plt.plot(upper_bounds)
        plt.show()
        s = mdp_solver(dist = u,
                       sigma = 0.4,
                       tau = 0.1,
                       p_A = 0,
                       p_D = 0,
                       N = 1000,
                       gamma = 0.8,
                       alpha = 0.15,
                       epsilon = 0.0001)
        plt.plot(s.R[-1, :])
        plt.show()
        plt.plot(s.R[1, :])
        plt.show()
        Q = s.run()
        # Q[state, policy]
        plt.plot(Q.argmax(axis=1)[1:-1])
        plt.show()
