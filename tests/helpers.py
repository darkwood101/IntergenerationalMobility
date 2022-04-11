import random
from aamodel.agent import privilege

# Helper functions for testing
class helpers:
    @staticmethod
    def a_dist_uniform():
        return random.uniform(0, 1)

    @staticmethod
    def a_dist_best():
        return 1.0

    @staticmethod
    def a_dist_worst():
        return 0.0

    @staticmethod
    def c_random():
        return random.choice([privilege.PRIVILEGED, privilege.NOT_PRIVILEGED])

    @staticmethod
    def sigma_tau_random():
        sigma = random.uniform(0, 1)
        tau = random.uniform(0, 1 - sigma)
        return sigma, tau

    @staticmethod
    def phi_0_random():
        return random.uniform(0, 1)

    @staticmethod
    def n_privileged_random(N):
        return random.randint(1, N)

    @staticmethod
    def p_A_random():
        return random.uniform(0, 1)

    @staticmethod
    def p_D_random():
        return random.uniform(0, 1)

    @staticmethod
    def N_random():
        return random.randint(1, 1000)
