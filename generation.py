from agent import agent, privilege


class generation:
    agents: list[agent]
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
                 agents: list[agent]):
        self.N = N
        self.sigma = sigma
        self.tau = tau
        self.phi_0 = phi_0
        self.p_A = p_A
        self.p_D = p_D
        self.a_dist = a_dist
        self.agents = agents \
                      if agents is not None \
                      else [agent(a_dist=self.a_dist,
                                  c=self.generate_privilege(),
                                  sigma=self.sigma,
                                  tau=self.tau,
                                  phi_0=self.phi_0,
                                  p_A=self.p_A,
                                  p_D=self.p_D) for _ in range(self.N)] 

        
    def generate_privilege(self) -> privilege:
        return privilege.NOT_PRIVILEGED \
               if random() <= self.phi_0 \
               else privilege.PRIVILEGED


    def step(self, theta_0, theta_1) -> tuple[generation, int]:
        new_agents = list[agent]
        n_privileged = 0
        for a in self.agents:
            new_agent, succeeded = a.step(theta_0, theta_1)
            self.n_successes += int(succeeded)
            n_privileged += int(new_agent.is_privileged())
            new_agents.append(new_agent)
        new_phi_0 = n_privileged / self.N

        new_generation = generation(N=self.N,
                                    sigma=self.sigma,
                                    tau=self.tau,
                                    phi_0=new_phi_0,
                                    p_A=self.p_A,
                                    p_D=self.p_D,
                                    a_dist=self.a_dist,
                                    agents=new_agents)
        payoff = n_successes / len(self.agents)
        return new_generation, payoff

