from generation import generation


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

