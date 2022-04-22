import unittest

from aamodel.solver import mdp_solver, \
                           uniform_distribution, \
                           normal_distribution, \
                           DISCRETIZATION
import matplotlib.pyplot as plt
import numpy as np


class solver_test(unittest.TestCase):
    def test_R(self):
        s = mdp_solver(dist = uniform_distribution,
                       sigma = 0.4,
                       tau = 0.1,
                       p_A = 0,
                       p_D = 0,
                       N = 2000,
                       gamma = 0.99,
                       alpha = 0.15,
                       epsilon = 0.0001)
        Q = s.run()
        # Q[state, policy]
        states = np.linspace(0, 1, 2000)
        policies = Q.argmax(axis=1)[1:]
        policies = policies * s.sigma / DISCRETIZATION
        plt.plot(states, policies)
        plt.show()

        """
        s = mdp_solver(dist = uniform_distribution,
                       sigma = 0.4,
                       tau = 0.1,
                       p_A = 0,
                       p_D = 0,
                       N = 1000,
                       gamma = 0.99,
                       alpha = 0.75,
                       epsilon = 0.0001)
        Q = s.run()
        # Q[state, policy]
        states = np.linspace(0, 1, 1000)
        policies_norm = Q.argmax(axis=1)[1:]
        policies_norm = policies_norm * s.sigma / DISCRETIZATION
        plt.plot(states, policies_norm)

        plt.show()
        
        

        # calculate the optimal theta_1 for the corresponding theta_0
        print("policy shape")
        print(len(policies), policies[0], policies[len(policies) - 1])

        theta_1 = s.corresponding_theta_1(policies)

        policy_differences = np.minimum(theta_1 - policies, s.sigma - policies)
        plt.plot(states, policy_differences)
        plt.show()
        """
