import unittest

from aamodel.solver import mdp_solver, uniform_dist
import matplotlib.pyplot as plt


class solver_test(unittest.TestCase):
    def test_R(self):
        u = uniform_dist()
        s = mdp_solver(dist = u,
                       sigma = 0.4,
                       tau = 0.1,
                       p_A = 0,
                       p_D = 0,
                       N = 1000,
                       gamma = 0.8,
                       alpha = 0.15,
                       epsilon = 0.0001)

        Q = s.run()
        plt.plot(Q.argmax(axis=1)[1:])
        plt.show()
