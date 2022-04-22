import unittest

from aamodel.solver import mdp_solver, uniform_distribution, DISCRETIZATION
import matplotlib.pyplot as plt
import numpy as np


class solver_test(unittest.TestCase):
    def test_R(self):
        s = mdp_solver(dist = uniform_distribution,
                       sigma = 0.4,
                       tau = 0.1,
                       p_A = 0,
                       p_D = 0,
                       N = 1000,
                       gamma = 0.99,
                       alpha = 0.15,
                       epsilon = 0.0001)
        Q = s.run()
        # Q[state, policy]
        states = np.linspace(0, 1, 1000)
        policies = Q.argmax(axis=1)[1:]
        policies = policies * s.sigma / DISCRETIZATION
        plt.plot(states, policies)
        plt.show()

        # calculate the optimal theta_1 for the corresponding theta_0
        print("policy shape")
        print(len(policies), policies[0], policies[len(policies) - 1])

        theta_1 = s.corresponding_theta_1(policies)

        policy_differences = np.minimum(theta_1 - policies, s.sigma - policies)
        plt.plot(states, policy_differences)
        plt.show()