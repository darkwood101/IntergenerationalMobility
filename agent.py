from random import random
from enum import Enum

class privilege(Enum):
    PRIVILEGED = 1,
    NOT_PRIVILEGED = 2

class agent:
    a: float
    a_dist: Callable[[], float]
    c: privilege
    sigma: float
    tau: float
    phi_0: float
    p_A: float
    p_D: float

    maybe_given: bool
    given: bool
    succeeded: bool
    success_prob: float

    # TODO: set default values of parameters
    def __init__(self,
                 a_dist: Callable[[], float]
                 c: privilege,
                 sigma: float,
                 tau: float,
                 phi_0: float,
                 p_A: float,
                 p_D: float):
        self.a_dist = a_dist
        self.a = a_dist()
        self.c = c
        
        # Assuming parameters were sanity-checked upstream
        self.sigma = sigma
        self.tau = tau
        self.phi_0 = phi_0
        self.p_A = p_A
        self.p_D = p_D

        self.success_prob = self.a * self.sigma + int(self.c) * self.tau
        assert 0 <= self.success_prob <= 1

    # Returns true if this agent is privileged
    def is_privileged(self) -> bool:
        return self.c == privilege.PRIVILEGED

    # Gives an opportunity to this agent if success probability is above
    # threshold. Returns `True` if the opportunity is given and the agent
    # succeeds, `False` otherwise.
    def maybe_give_opportunity(self, theta_0, theta_1) -> bool:
        threshold = theta_1 if self.is_privileged() else theta_0

        # Mark that this agent was considered for an opportunity
        self.maybe_given = True

        # See if the agent exceeds the threshold and succeeds
        self.given = self.success_prob >= threshold
        self.succeeded = self.given and random() <= self.success_prob
        
        # If succeeded, moves up to privileged group, otherwise stays the same
        if self.succeeded:
            self.c = privilege.PRIVILEGED

        return self.succeeded

    # Returns the offspring of this agent, computed according to the model
    # Can only be called after opportunity allocation
    def produce_offspring(self) -> agent:
        assert self.maybe_given, "Cannot produce offspring until maybe given"

        new_c = privilege
        if self.is_privileged():
            new_c = privilege.PRIVILEGED \
                    if (random() <= self.p_A) and (random() > self.phi_0) \
                    else privilege.NOT_PRIVILEGED
        else:
            new_c = privilege.NOT_PRIVILEGED \
                    if (random() <= self.p_D) and (random() <= self.phi_0) \
                    else privilege.PRIVILEGED

        return agent(a_dist=self.a_dist,
                     c=new_c,
                     sigma=self.sigma,
                     tau=self.tau,
                     phi_0=self.phi_0,
                     p_A=self.p_A,
                     p_D=self.p_D)

    # Considers giving the opportunity to this agent. Returns the agent's
    # offspring, and a bool indicating whether the agent succeeded
    def step(self, theta_0, theta_1) -> tuple[agent, bool]:
        succeeded = self.maybe_give_opportunity(theta_0, theta_1)
        new_agent = self.produce_offspring()
        return new_agent, succeeded

