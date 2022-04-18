from aamodel.generation import generation
from typing import List, Callable, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod
from math import isclose

DISCRETIZATION = 100

counttt = 0

class distribution(ABC):
    @abstractmethod
    def sample(self) -> float:
        pass

    @abstractmethod
    def get_theta_1(self, theta_0, sigma, tau, alpha) -> float:
        pass

    @abstractmethod
    def expected_priv_payoff(self, theta_1, phi_0, sigma, tau) -> float:
        pass

    @abstractmethod
    def expected_unpriv_payoff(self, theta_0, phi_0, sigma, tau) -> float:
        pass

    @abstractmethod
    def expected_payoff(self, theta_0, N, n_unpriv, sigma, tau, alpha):
        pass

    @abstractmethod
    def action_bounds(self, phi_0, sigma, alpha) -> Tuple[float, float]:
        pass


class uniform_dist(distribution):
    def sample(self):
        return random.uniform(0, 1)

    # Compute `theta_1` given a `theta_0`
    def get_theta_1(self, theta_0, phi_0, sigma, tau, alpha):
        if isclose(phi_0, 1.0):
            return tau + sigma / 2

        return (sigma * (1.0 - alpha) + (1.0 - phi_0) * tau - phi_0 * theta_0) / \
               (1.0 - phi_0)

    # Compute expected payoff from privileged people for a given threshold
    # `theta_1`
    def expected_priv_payoff(self, theta_1, phi_0, sigma, tau):
        if theta_1 < tau:
            return (1 - phi_0) * (tau + sigma) / 2
        if theta_1 - tau > sigma:
            return 0
        return ((1 - phi_0) / (2 * sigma)) * ((sigma + tau) ** 2 - theta_1 ** 2)

     
    # Compute expected payoff from unprivileged people for a given threshold
    # `theta_0`
    def expected_unpriv_payoff(self, theta_0, phi_0, sigma, tau):
        if theta_0 > sigma:
            return 0
        return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)


    # Compute expected payoff from everyone for given thresholds
    # `theta_0` and `theta_1`
    def expected_payoff(self, theta_0, N, n_unpriv, sigma, tau, alpha):
        phi_0 = n_unpriv / N
        theta_1 = self.get_theta_1(theta_0, phi_0, sigma, tau, alpha)

        if isclose(phi_0, 1.0):
            return (1 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)

        return (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2) + \
               ((1.0 - phi_0) / (2 * sigma)) + ((sigma + tau) ** 2 - theta_1 ** 2)
        """
        print(n_unpriv)
        print(theta_1)
        print(theta_0)
        print(self.expected_priv_payoff(theta_1, phi_0, sigma, tau)) 
        """
        assert 0 <= self.expected_priv_payoff(theta_1, phi_0, sigma, tau) <= 1
        assert 0 <= self.expected_unpriv_payoff(theta_0, phi_0, sigma, tau) <= 1
        assert 0 <= self.expected_priv_payoff(theta_1, phi_0, sigma, tau) + \
                    self.expected_unpriv_payoff(theta_0, phi_0, sigma, tau) <= 1
        return self.expected_priv_payoff(theta_1, phi_0, sigma, tau) + \
               self.expected_unpriv_payoff(theta_0, phi_0, sigma, tau)

    # 
    def action_bounds(self, phi_0, sigma, alpha):
        if isclose(phi_0, 0.0):
            return 0, 1
        lower = max(sigma * (1 - alpha / phi_0), 0)
        upper = min(sigma * (1 - alpha) / phi_0, 1)
        assert lower <= upper
        return lower, upper
        return lower, upper


class statistics:
    def __init__(self, n_episodes: int):
        self.rewards = np.zeros(n_episodes)
        self.lengths = np.zeros(n_episodes)


class mdp_solver:
    gen: generation
    Q: List[List[float]]
    R: List[List[float]]
    T: List[List[float]]
    dist: distribution
    sigma: float
    tau: float
    p_A: float
    p_D: float
    N: int
    gamma: float
    alpha: float
    epsilon: float


    def __init__(self,
                 dist: distribution,
                 sigma: float,
                 tau: float,
                 p_A: float,
                 p_D: float,
                 N: int,
                 gamma: float,
                 alpha: float,
                 epsilon: float):
        self.gen = generation(a_dist = dist.sample,
                              sigma = sigma,
                              tau = tau,
                              n_privileged = None,
                              p_A = p_A,
                              p_D = p_D,
                              N = N)
        self.Q = np.zeros((N + 1, DISCRETIZATION + 1), dtype = float)
        self.dist = dist
        self.sigma = sigma
        self.tau = tau
        self.p_A = p_A
        self.p_D = p_D
        self.N = N
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.init_R_S()
                

    def init_R_S(self):
        self.R = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = float)
        self.S = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = np.int32)
        for s in range(self.N + 1):
            phi_0 = s / self.N
            for a in range(DISCRETIZATION + 1):
                theta_0 = a / DISCRETIZATION
                self.R[s, a] = self.dist.expected_payoff(theta_0,
                                                         self.N,
                                                         s,
                                                         self.sigma,
                                                         self.tau,
                                                         self.alpha)
                if theta_0 > self.sigma:
                    phi_0_post = phi_0
                else:
                    phi_0_post = phi_0 - (phi_0 / (2 * self.sigma)) * \
                                         (self.sigma ** 2 - theta_0 ** 2)
                """
                self.S[s, a] = int((phi_0_post * (1 - self.p_D) + \
                               (phi_0_post ** 2) * self.p_D + \
                               (1 - phi_0_post) * self.p_A * phi_0_post) * \
                               self.N)
                """
                self.S[s, a] = phi_0 - (phi_0 / (2 * self.sigma)) * \
                                       (self.sigma ** 2 - theta_0 ** 2)


    def run(self):
        while True:
            Q_new = np.zeros((self.N + 1, DISCRETIZATION + 1), dtype = float)
            for s in range(self.N + 1):
                phi_0 = s / self.N
                lower_action, upper_action = \
                        self.dist.action_bounds(phi_0, self.sigma, self.alpha)
                lower_action = int(lower_action * DISCRETIZATION)
                upper_action = int(upper_action * DISCRETIZATION)

                for a in range(lower_action, upper_action):
                    s_new = self.S[s, a]
                    if self.R[s, a] > self.alpha:
                        print(self.R[s, a])
                    Q_new[s, a] = self.R[s, a] + \
                                  self.gamma * np.max(self.Q[s_new])
            max_e = np.max(np.abs(self.Q - Q_new))
            self.Q = Q_new
            if max_e < self.epsilon:
                return Q_new


class solver:
    gen: generation
    Q: List[List[float]]
    dist: distribution
    gamma: float
    alpha: float
    epsilon: float
    delta: float
    lrate: float

    def __init__(self,
                 dist: distribution,
                 sigma: float,
                 tau: float,
                 p_A: float,
                 p_D: float,
                 N: int,
                 gamma: float,
                 alpha: float,
                 epsilon: float,
                 delta: float,
                 lrate: float):
        self.gen = generation(a_dist = dist.sample,
                              sigma = sigma,
                              tau = tau,
                              n_privileged = None,
                              p_A = p_A,
                              p_D = p_D,
                              N = N)
        self.Q = np.zeros((N + 1, DISCRETIZATION + 1), dtype = float)
        self.dist = dist
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.delta = delta
        self.lrate = lrate

    def run(self, n_episodes):
        stats = statistics(n_episodes)

        for episode in range(n_episodes):
            self.gen.reset()
            old_s = self.gen.n_privileged
            print("ok here we go", episode)
            count = 0
            while True:
                count += 1
                theta_0 = None
                if random.uniform(0, 1) < self.epsilon:
                    action = random.randint(0, DISCRETIZATION)
                else:
                    action = np.argmax(self.Q[old_s])
                theta_0 = action / DISCRETIZATION
                theta_1 = self.dist.get_theta_1(theta_0, self)
                payoff = self.gen.step(theta_0, theta_1) / self.gen.N
                new_s = self.gen.n_privileged

                incr = payoff + \
                       self.gamma * np.max(self.Q[new_s]) - \
                       self.Q[old_s, action]
                incr *= self.lrate
                self.Q[old_s, action] += incr
                
                stats.rewards[episode] += payoff
                #if abs(incr) < self.delta:
                if count > 500:
                    stats.lengths[episode] = count
                    break

                old_s = new_s

        return self.Q, stats


"""
class world:
    t: int
    N: int
    alpha: float
    sigma: float
    tau: float
    gamma: float
    p_A: float
    p_D: float
    generations: list[generation]

    # TODO: set default values of parameters
    def __init__(self, N, alpha, sigma, tau, phi_0, gamma, p_A, p_D):
        self.t = 0

        assert N > 0
        self.N = N
        
        assert 0 < alpha < 1
        self.alpha = alpha
        
        assert (sigma > 0) and (tau > 0) and (sigma + tau <= 1)
        self.sigma = sigma
        self.tau = tau

        assert 0 < gamma <= 1
        self.gamma = gamma

        assert (0 <= p_a <= 1) and (0 <= p_D <= 1)
        self.p_A = p_A
        self.p_D = p_D

        assert 0 <= phi_0 <= 1
        # TODO: pass more arguments here after finishing
        # the generation constructor
        self.generations = [generation(...)]

    def get_time(self):
        return self.t

    def step(self):
        raise NotImplemented
"""

