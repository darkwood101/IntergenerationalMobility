from cmath import isnan
from math import isclose, nan, inf, isnan
import numpy as np
from scipy.stats import norm

DISCRETIZATION = 100

class uniform_distribution:
    @staticmethod
    def allowed_actions(phi_0, sigma, alpha):
        if isclose(phi_0, 0.0):
            lower = 0
            upper = sigma
        else:
            lower = max(sigma * (1.0 - alpha / phi_0), 0)
            upper = min(sigma * (1.0 - alpha) / phi_0, sigma)
        return lower, upper

    @staticmethod
    def get_payoff(theta_0, phi_0, sigma, tau, alpha):
        assert sigma >= theta_0

        if isclose(phi_0, 1.0):
            theta_1 = sigma + tau
        else:
            theta_1 = (sigma * (1.0 - alpha) + \
                      (1.0 - phi_0) * tau - phi_0 * theta_0) / (1.0 - phi_0)

        if tau + sigma < theta_1:
            return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)

        return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2) + \
               ((1.0 - phi_0) / (2 * sigma)) * ((sigma + tau) ** 2 - theta_1 ** 2)

def quantile(x, sigma):
    # returns quanitle value between -inf and inf
    # we want to cap it to 0 and 1
    quantile = norm.ppf(x, 0.5, 1/sigma) 
    
    if quantile > 1 or x >= 1:
        return 1
    if quantile < 0 or x <= 0:
        return 0

    return quantile

def normal_allowed_actions(phi_0, sigma, alpha):
    if isclose(phi_0, 0.0):
        lower = 0
        upper = sigma
    else:
       
        lower = sigma * quantile(1.0 - (alpha/phi_0), sigma)

        upper = sigma * quantile((1.0 - alpha)/phi_0, sigma)
    if lower > upper:
        print(lower, upper, (alpha - 1)/phi_0, 1 - (alpha/phi_0))
        #upper = min(sigma * (1.0 - alpha) / phi_0, sigma)

    

    
    return lower, upper

def normal_get_payoff(theta_0, phi_0, sigma, tau, alpha):
    assert sigma >= theta_0

    if isclose(phi_0, 1.0):
        theta_1 = sigma + tau
    else:
        theta_1 = (sigma * (1.0 - alpha) + (1.0 - phi_0) * tau - phi_0 * theta_0) / \
                  (1.0 - phi_0)

    if tau + sigma < theta_1:
        return phi_0 * (1 - norm.cdf(theta_0/sigma, 0.5, sigma/10)) * \
            (sigma * quantile(0.5 * (1 - norm.cdf(theta_0/sigma, 0.5, sigma/10) + norm.cdf(theta_0/sigma, 0.5, sigma/10)), sigma))

    return phi_0 * (1 - norm.cdf(theta_0/sigma, 0.5, sigma/10)) * \
            (sigma * quantile( (0.5 * (1 - norm.cdf(theta_0/sigma, 0.5, sigma/10))) + norm.cdf(theta_0/sigma, 0.5, sigma/10), sigma)) + \
            (1 - phi_0) * (1 - norm.cdf((theta_1 - tau)/sigma)) * \
            (sigma * quantile((0.5 *(1 - norm.cdf((theta_1 - tau)/sigma))) + norm.cdf((theta_1 - tau)/sigma, 0.5, sigma/10), sigma))

    

class mdp_solver:
    def __init__(self,
                 dist,
                 sigma,
                 tau,
                 p_A,
                 p_D,
                 N,
                 gamma,
                 alpha,
                 epsilon):
        self.dist = dist
        self.sigma = sigma
        self.tau = tau
        self.p_A = p_A
        self.p_D = p_D
        self.N = N
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon

        self.Q = np.zeros((N + 1, DISCRETIZATION + 1), dtype = float)
        self.R = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = float)
        self.S = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)
        self.mask = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)

        for s in range(self.N + 1):
            phi_0 = s / self.N
<<<<<<< HEAD
            lower, upper = normal_allowed_actions(phi_0 = phi_0,
                                           sigma = self.sigma,
                                           alpha = self.alpha)
            
           
=======
            lower, upper = dist.allowed_actions(phi_0 = phi_0,
                                                sigma = self.sigma,
                                                alpha = self.alpha)
>>>>>>> main
            assert 0 <= lower <= self.sigma
            assert 0 <= upper <= self.sigma
            assert lower <= upper
            lower = int(lower * DISCRETIZATION / self.sigma)
            upper = int(upper * DISCRETIZATION / self.sigma)
            self.mask[s, lower : upper + 1] = 1
            for a in range(lower, upper + 1):
                theta_0 = a * self.sigma / DISCRETIZATION
<<<<<<< HEAD
                self.R[s, a] = normal_get_payoff(theta_0 = theta_0,
                                          phi_0 = phi_0,
                                          sigma = self.sigma,
                                          tau = self.tau,
                                          alpha = self.alpha)
=======
                self.R[s, a] = dist.get_payoff(theta_0 = theta_0,
                                               phi_0 = phi_0,
                                               sigma = self.sigma,
                                               tau = self.tau,
                                               alpha = self.alpha)
>>>>>>> main
                # Ok this is passing which is good
                if self.R[s, a] <= 0 or self.R[s, a] >= 1:
                    print(theta_0, a, self.R[s, a])
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
