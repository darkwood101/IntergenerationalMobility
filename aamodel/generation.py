from aamodel.agent import agent, privilege
from typing import List, Callable, Tuple
from random import random, randint

class generation:
    # These are provided at construction time
    # They are used for _all_ agents in the generation
    a_dist: Callable[[], float]     # Distribution of abilities
    sigma: float                    # Ability multiplier
    tau: float                      # Privilege multiplier
    n_privileged: int               # Number of privileged agents in this
                                    # generation (used to find `phi_0`)
    p_A: float                      # Probability of movement for privileged
                                    # agents
    p_D: float                      # Probability of movement for unprivileged
                                    # agents
    N: int                          # Number of agents in this generation

    # These are derived during initialization
    agents: List[agent]             # List of agents in this generation


    def __init__(self,
                 a_dist: Callable[[], float],
                 sigma: float,
                 tau: float,
                 n_privileged: int,
                 p_A: float,
                 p_D: float,
                 N: int):
        self.a_dist = a_dist
        self.sigma = sigma
        self.tau = tau
        if n_privileged is None:
            self.n_privileged = randint(0, N)
        else:
            self.n_privileged = n_privileged
        self.p_A = p_A
        self.p_D = p_D
        self.N = N

        p_agents = [agent(a_dist = self.a_dist,
                          c = privilege.PRIVILEGED,
                          sigma = self.sigma,
                          tau = self.tau,
                          p_A = self.p_A,
                          p_D = self.p_D) \
                    for _ in range(self.n_privileged)]
        np_agents = [agent(a_dist = self.a_dist,
                           c = privilege.NOT_PRIVILEGED,
                           sigma = self.sigma,
                           tau = self.tau,
                           p_A = self.p_A,
                           p_D = self.p_D) \
                     for _ in range(self.N - self.n_privileged)]
        self.agents = p_agents + np_agents

    
    @property
    def phi_0(self):
        return 1 - self.n_privileged / self.N
    

    # Evolves the generation according to the model.
    # Allocates opportunities to agents in accordance with the provided
    # thresholds `theta_0` and `theta_1`. Evolves this generation into the next
    # generation.
    # Returns the number of successes in this generation (can be used by the
    # caller to get the payoff).
    def step(self, theta_0: float, theta_1: float) -> int:
        n_successes = 0
        old_phi_0 = self.phi_0

        # Reset the number of privileged agents and count them again
        # (will also reset `phi_0`)
        self.n_privileged = 0

        # Update statistics by iterating through old agents
        for a in self.agents:
            n_successes += int(a.step(theta_0, theta_1, old_phi_0))
            self.n_privileged += int(a.is_privileged())

        return n_successes

    
    # Reset the generation to a new state
    def reset(self):
        self.n_privileged = randint(0, self.N)
        for i in range(0, self.n_privileged):
            self.agents[i].c = True
        for i in range(0, self.N - self.n_privileged):
            self.agents[i].c = False

