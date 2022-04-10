from agent import agent


class generation:
    agents: list[agent]
    sigma: float
    n_successes: float
    payoff: float

    # TODO: see comment in `step()`
    def __init__(self,
                 sigma: float,
                 tau: float,
                 phi_0: float):
        raise NotImplemented

    def step(self, theta_0, theta_1) -> tuple[generation, int]:
        new_agents = list[agent]
        for a in self.agents:
            new_agent, succeeded = a.step(theta_0, theta_1)
            n_successes += int(succeeded)
            new_agents.append(new_agent)

        # TODO: need a way to construct a new generation from an existing list
        # of agents
        # have an argument in the constructor, if it's None, then generate
        # a brand new generation, otherwise construct from the list
        new_generation = ...
        payoff = n_successes / len(self.agents)
        return new_generation, payoff

