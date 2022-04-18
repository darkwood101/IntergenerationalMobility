from math import isclose
import numpy as np

DISCRETIZATION = 1000

def allowed_actions(phi_0, sigma, alpha):
    if isclose(phi_0, 0.0):
        lower = 0
        upper = sigma
    else:
        lower = max(sigma * (1.0 - alpha / phi_0), 0)
        upper = min(sigma * (1.0 - alpha) / phi_0, sigma)
    return lower, upper

def get_payoff(theta_0, phi_0, sigma, tau, alpha):
    assert sigma >= theta_0

    if isclose(phi_0, 1.0):
        theta_1 = sigma + tau
    else:
        theta_1 = (sigma * (1.0 - alpha) + (1.0 - phi_0) * tau - phi_0 * theta_0) / \
                  (1.0 - phi_0)

    if tau + sigma < theta_1:
        return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)

    return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2) + \
           ((1.0 - phi_0) / (2 * sigma)) * ((sigma + tau) ** 2 - theta_1 ** 2)

class mdp_solver:
    def __init__(self,
                 sigma,
                 tau,
                 p_A,
                 p_D,
                 N,
                 gamma,
                 alpha,
                 epsilon):
        self.Q = np.zeros((N + 1, DISCRETIZATION + 1), dtype = float)
        self.sigma = sigma
        self.tau = tau
        self.p_A = p_A
        self.p_D = p_D
        self.N = N
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.R = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = float)
        self.S = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)
        self.mask = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)

        for s in range(self.N + 1):
            phi_0 = s / self.N
            lower, upper = allowed_actions(phi_0 = phi_0,
                                           sigma = self.sigma,
                                           alpha = self.alpha)
            assert 0 <= lower <= self.sigma
            assert 0 <= upper <= self.sigma
            assert lower <= upper
            lower = int(lower * DISCRETIZATION / self.sigma)
            upper = int(upper * DISCRETIZATION / self.sigma)
            self.mask[s, lower : upper + 1] = 1
            for a in range(lower, upper + 1):
                theta_0 = a * self.sigma / DISCRETIZATION
                self.R[s, a] = get_payoff(theta_0 = theta_0,
                                          phi_0 = phi_0,
                                          sigma = self.sigma,
                                          tau = self.tau,
                                          alpha = self.alpha)
                # Ok this is passing which is good
                assert 0 <= self.R[s, a] <= self.alpha
                phi_0_post = phi_0 - (phi_0 / (2 * sigma)) * \
                                     (sigma ** 2 - theta_0 ** 2)
                phi_0_new = phi_0_post * (1.0 - self.p_D) + \
                            (phi_0_post ** 2) * self.p_D + \
                            (1 - phi_0_post) * self.p_A * phi_0_post
                s_new = int(phi_0_new * self.N)
                self.S[s, a] = s_new
                
    def run(self):
        # MEGA HACKY TRICK
        #
        # We make a `mask` array where allowed state-policy pairs have value 1,
        # and disallowed pairs have value 0
        #
        # Then, after we're done broadcasting, we multiply the result by this
        # `mask` array, and we get rid of anything that's not allowed
        #
        # 10 seconds for gamma == 0.99 convergence
        while True:
            Q_new = (self.R + self.gamma * self.Q.max(axis = 1)[self.S]) * \
                    self.mask
            max_e = np.max(np.abs(self.Q - Q_new))
            print("diff:", max_e)
            self.Q = Q_new
            if max_e < self.epsilon:
                break

        return self.Q
