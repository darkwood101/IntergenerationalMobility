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

        # Set everything to -1 to catch bugs
        self.R = -np.ones((self.N + 1, DISCRETIZATION + 1), dtype = float)
        self.S = -np.ones((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)

        for s in range(self.N + 1):
            phi_0 = s / self.N
            lower, upper = allowed_actions(phi_0 = phi_0,
                                           sigma = self.sigma,
                                           alpha = self.alpha)
            assert 0 <= lower <= self.sigma
            assert 0 <= upper <= self.sigma
            assert lower <= upper
            lower = int(lower * DISCRETIZATION)
            upper = int(upper * DISCRETIZATION)
            for a in range(lower, upper + 1):
                theta_0 = a / DISCRETIZATION
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
        while True:
            Q_new = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = float)
            for s in range(self.N + 1):
                phi_0 = s / self.N
                lower, upper = allowed_actions(phi_0 = phi_0,
                                               sigma = self.sigma,
                                               alpha = self.alpha)
                assert 0 <= lower <= self.sigma
                assert 0 <= upper <= self.sigma
                assert lower <= upper
                lower = int(lower * DISCRETIZATION)
                upper = int(upper * DISCRETIZATION)
                for a in range(lower, upper + 1):
                    s_new = self.S[s, a]
                    # This also good
                    assert 0 <= self.R[s, a] <= self.alpha
                    Q_new[s, a] = self.R[s, a] + \
                                  self.gamma * np.max(self.Q[s_new])
            max_e = np.max(np.abs(self.Q - Q_new))
            print(max_e)
            self.Q = Q_new
            if max_e < self.epsilon:
                return Q_new
