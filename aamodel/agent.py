from random import random
from enum import Enum
from typing import Callable, Type, Tuple

class privilege(Enum):
    PRIVILEGED = 1,
    NOT_PRIVILEGED = 2

class agent:
    # These are provided at construction time
    a_dist: Callable[[], float]     # Distribution of abilities
    c: privilege                    # Privilege (circumstance from the paper)
    sigma: float                    # Ability multiplier
    tau: float                      # Privilege multiplier
    phi_0: float                    # Fraction of unprivileged agents in the
                                    # generation to which this agent belongs
    p_A: float                      # Probability of movement for privileged
                                    # agents
    p_D: float                      # Probability of movement for unprivileged
                                    # agents

    # These are derived during initialization
    a: float                        # Ability
    maybe_given: bool               # `True` iff this agent was already
                                    # considered for opportunity allocation
    given: bool                     # `True` iff this agent was allocated an
                                    # opportunity
    succeeded: bool                 # `True` iff this agent was allocated an
                                    # opportunity AND succeeded
    success_prob: float             # Success probability for this agent


    # TODO: set default values of parameters
    def __init__(self,
                 a_dist: Callable[[], float],
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

        self.success_prob = self.a * self.sigma + \
                            int(self.is_privileged()) * self.tau
        assert 0 <= self.success_prob <= 1

        self.maybe_given = False
        self.given = False
        self.succeeded = False


    # Returns true iff this agent is privileged.
    def is_privileged(self) -> bool:
        return self.c == privilege.PRIVILEGED


    # Gives an opportunity to this agent if success probability is above
    # threshold. Returns `True` if the opportunity is given and the agent
    # succeeds, `False` otherwise.
    def maybe_give_opportunity(self, theta_0, theta_1) -> bool:
        threshold = theta_1 if self.is_privileged() else theta_0

        # Mark that this agent was considered for an opportunity
        self.maybe_given = True

        # See if the agent exceeds the threshold (i.e. gets the opportunity)
        # and succeeds
        self.given = self.success_prob >= threshold
        self.succeeded = self.given and random() <= self.success_prob
        
        # If succeeded, moves up to privileged group, otherwise stays the same
        if self.succeeded:
            self.c = privilege.PRIVILEGED

        return self.succeeded


    # Returns the offspring of this agent, computed according to the model.
    # Can only be called after opportunity allocation.
    def produce_offspring(self) -> 'agent':
        assert self.maybe_given, "Cannot produce offspring maybe given"

        new_c: privilege
        if self.is_privileged():
            new_c = privilege.NOT_PRIVILEGED \
                    if (random() <= self.p_A) and (random() <= self.phi_0) \
                    else privilege.PRIVILEGED
        else:
            new_c = privilege.PRIVILEGED \
                    if (random() <= self.p_D) and (random() > self.phi_0) \
                    else privilege.NOT_PRIVILEGED

        # Everything is inherited from this agent, except possibly circumstance
        return agent(a_dist = self.a_dist,
                     c = new_c,
                     sigma = self.sigma,
                     tau = self.tau,
                     phi_0 = self.phi_0,
                     p_A = self.p_A,
                     p_D = self.p_D)


    # Evolves the agent according to the model.
    # Considers giving the opportunity to this agent. Returns the agent's
    # offspring, and a bool indicating whether the agent succeeded.
    def step(self, theta_0, theta_1) -> Tuple['agent', bool]:
        succeeded = self.maybe_give_opportunity(theta_0, theta_1)
        new_agent = self.produce_offspring()
        return new_agent, succeeded

