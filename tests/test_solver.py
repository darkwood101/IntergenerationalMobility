import unittest

from aamodel.solver import mdp_solver
from aamodel.uniform_distribution import uniform_distribution
from aamodel.normal_distribution import normal_distribution
import matplotlib.pyplot as plt
import numpy as np


class solver_test(unittest.TestCase):
    def test_R(self):
        s_u = mdp_solver(dist = uniform_distribution(0, 1),
                         sigma = 0.4,
                         tau = 0.1,
                         p_A = 0,
                         p_D = 0,
                         N = 2000,
                         gamma = 0.8,
                         alpha = 0.15)
        states_u, theta_0_u, theta_1_u = s_u.run()

        s_n = mdp_solver(dist = normal_distribution(0.5, 0.05),
                         sigma = 0.4,
                         tau = 0.1,
                         p_A = 0,
                         p_D = 0,
                         N = 2000,
                         gamma = 0.8,
                         alpha = 0.15)
        states_n, theta_0_n, theta_1_n = s_n.run()

        plt.plot(states_u[1:], theta_0_u[1:])
        plt.plot(states_n[1:], theta_0_n[1:])
        plt.show()


        policy_diff_u = np.minimum(theta_1_u - theta_0_u, s_u.sigma - theta_0_u)
        policy_diff_n = np.minimum(theta_1_n - theta_0_n, s_n.sigma - theta_0_n)
        plt.plot(states_u[1:], policy_diff_u[1:])
        plt.plot(states_n[1:], policy_diff_n[1:])
        plt.show()
