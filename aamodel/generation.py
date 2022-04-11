from aamodel.agent import agent, privilege
from typing import List, Callable, Tuple
from random import random, shuffle

class generation:
    # These are provided at construction time
    # They are used for _all_ agents in the generation
    a_dist: Callable[[], float]     # Distribution of abilities
    sigma: float                    # Ability multiplier
    tau: float                      # Privilege multiplier
    phi_0: float                    # Fraction of unprivileged agents in this
                                    # generation
    p_A: float                      # Probability of movement for privileged
                                    # agents
    p_D: float                      # Probability of movement for unprivileged
                                    # agents
    N: int                          # Number of agents in this generation
    agents: List[agent]             # Optional: List of agents from the previous
                                    # generation, used to construct the next
                                    # generation

    # These are derived during initialization
    n_successes: float              # Number of successful agents in this
                                    # generation
    payoff: float                   # Undiscounted payoff from this generation


    def __init__(self,
                 a_dist: Callable[[], float],
                 sigma: float,
                 tau: float,
                 phi_0: float,
                 p_A: float,
                 p_D: float,
                 N: int,
                 agents: List[agent]):
        self.a_dist = a_dist
        self.sigma = sigma
        self.tau = tau
        self.p_A = p_A
        self.p_D = p_D
        self.N = N

        if agents:
            # In this case, we are constructing from the previous generation.
            # It is the caller's responsibility to ensure that `phi_0` and
            # `agents` are consistent with each other.
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

    
    # Evolves the generation according to the model.
    # Allocates opportunities to agents in accordance with the provided
    # thresholds `theta_0` and `theta_1`. Returns the next generation, and the
    # undiscounted payoff from this generation.
    def step(self, theta_0, theta_1) -> Tuple['generation', float]:
        new_agents: List[agent] = []
        # Need to count the number of privileged agents for the new `phi_0`
        n_privileged = 0

        # Update statistics by iterating through old agents, and construct
        # the new agent list
        for a in self.agents:
            new_agent, succeeded = a.step(theta_0, theta_1)
            self.n_successes += int(succeeded)
            n_privileged += int(new_agent.is_privileged())
            new_agents.append(new_agent)
        new_phi_0 = 1 - n_privileged / self.N

        # Construct the new generation
        new_generation = generation(N = self.N,
                                    sigma = self.sigma,
                                    tau = self.tau,
                                    phi_0 = new_phi_0,
                                    p_A = self.p_A,
                                    p_D = self.p_D,
                                    a_dist = self.a_dist,
                                    agents = new_agents)
        self.payoff = self.n_successes / len(self.agents)
        return new_generation, self.payoff

