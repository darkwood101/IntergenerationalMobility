from aamodel.agent import agent, privilege
from typing import List, Callable, Tuple
from random import random, shuffle

class generation:
    agents: List[agent]
    N: int
    sigma: float
    tau: float
    p_A: float
    p_D: float
    phi_0: float
    a_dist: Callable[[], float]

    n_successes: float
    payoff: float

    def __init__(self,
                 N: int,
                 sigma: float,
                 tau: float,
                 phi_0: float,
                 p_A: float,
                 p_D: float,
                 a_dist: Callable[[], float],
                 agents: List[agent]):
        self.N = N
        self.sigma = sigma
        self.tau = tau
        self.p_A = p_A
        self.p_D = p_D
        self.a_dist = a_dist

        if agents:
            self.agents = agents
            self.phi_0 = phi_0
        else:
            # This is a bit janky, might have to change it.
            # Basically, if we are creating a brand new generation and we
            # specify `phi_0`, we might not be able to achieve exactly `phi_0`,
            # because of rounding.
            # The solution is to compute the rounded number of privileged
            # agents, and then use that to compute the exact `self.phi_0`
            #
            # TLDR: If `agents == None`, then `self.phi_0 != phi_0`
            n_unprivileged = round(phi_0 * self.N)
            self.phi_0 = n_unprivileged / self.N
            p_agents = [agent(a_dist = self.a_dist,
                              c = privilege.PRIVILEGED,
                              sigma = self.sigma,
                              tau = self.tau,
                              phi_0 = self.phi_0,
                              p_A = self.p_A,
                              p_D = self.p_D) \
                        for _ in range(self.N - n_unprivileged)]
            np_agents = [agent(a_dist = self.a_dist,
                               c = privilege.NOT_PRIVILEGED,
                               sigma = self.sigma,
                               tau = self.tau,
                               phi_0 = self.phi_0,
                               p_A = self.p_A,
                               p_D = self.p_D) \
                         for _ in range(n_unprivileged)]
            self.agents = p_agents + np_agents
            shuffle(self.agents)

        self.n_successes = 0
        self.payoff = 0


    def step(self, theta_0, theta_1) -> Tuple['generation', int]:
        new_agents: List[agent] = []
        n_privileged = 0
        for a in self.agents:
            new_agent, succeeded = a.step(theta_0, theta_1)
            self.n_successes += int(succeeded)
            n_privileged += int(new_agent.is_privileged())
            new_agents.append(new_agent)
        new_phi_0 = 1 - n_privileged / self.N

        new_generation = generation(N = self.N,
                                    sigma = self.sigma,
                                    tau = self.tau,
                                    phi_0 = new_phi_0,
                                    p_A = self.p_A,
                                    p_D = self.p_D,
                                    a_dist = self.a_dist,
                                    agents = new_agents)
        payoff = self.n_successes / len(self.agents)
        return new_generation, payoff
