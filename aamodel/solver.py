import numpy as np


class mdp_solver:
    """
    This class fins the optimal policies for every state of the system.
    The state is defined as `phi_0`, the fraction of the population who are
    unprivileged.
    The policy is `theta_0`, the threshold for opportunity allocation for the
    unprivileged population (the threshold for the privileged population
    `theta_1` is determined from `theta_0`).

    Since our state and policy spaces are discretized, internally we use
    integers to label them. That is, a state `i` refers to `phi_0 = i / N`,
    while a policy `j` refers to `theta_0 = j / discretization`.

    Attributes
    ----------
    dist
        The distribution of individuals' abilities. We currently support two
        distributions: `uniform_distribution` and `normal_distribution`.
        Every distribution has to implement the following methods:
        `allowed_actions`, `theta_1_from_theta_0`, `get_payoff`, and
        `phi_0_post`. For descriptions of these methods, see
        `uniform_distribution`.
    sigma : float
        The ability multiplier. Always has to be in [0, 1].
    tau : float
        The privilege multiplier. Always has to be in [0, 1]. In addition, we
        need to have `sigma + tau <= 1`.
    p_A : float
        Probability of privilege redistribution for the privileged group.
        Always has to be in [0, 1].
    p_D : float
        Probability of privilege redistribution for the unprivileged group.
        Always has to be in [0, 1].
    N : int
        Number of agents. This will determine `phi_0` discretization.
    gamma : float
        Reward discount factor. Always has to be in (0, 1).
    alpha : float
        Fraction of population that can receive an opportunity in a single
        generation. That is, we assume that we can give no more than
        `alpha * N` opportunities.
    discretization : int
        Discretization of the policy space. By default, we choose 2000. A larger
        number will increase running time. Note that policies are not
        discretized by dividing [0, 1] into `discretization` pieces, but by
        dividing [0, `sigma`] into `discretization` pieces. This is because
        the largest possible policy is `sigma`.
    Q : numpy.ndarray
        A float array of shape `(N + 1, discretization + 1)`. Here, we store
        infinite-horizon rewards for state-policy pairs during the learning
        process. The entry `Q[i, j]` gives the infinite-horizon reward for
        taking policy `j` in state `i`.
    R : numpy.ndarray
        A float array of shape `(N + 1, discretization + 1)`. This array stores
        immediate rewards for state-policy pairs. The entry `Q[i, j]` gives the
        immediate undiscounted reward for taking policy `j` in state `i`.
    S : numpy.ndarray
        An integer array of shape `(N + 1, discretization + 1)`. This array
        stores the transition states for state-policy pairs. The entry `Q[i, j]`
        gives the state to which the system transitions when taking policy `j`
        in state `i`. The states are stored as integers in [0, N].
    mask : numpy.ndarray
        A boolean array of shape `(N + 1, discretization + 1)`. An entry
        `Q[i, j]` is `True` iff policy `j` is allowed to be taken from state
        `i`. A policy would not be allowed when it would lead to the total
        amount of allocated opportunities being strictly less or more than
        `alpha`.
    theta_0 : numpy.ndarray
        A float array of shape `(N + 1,)`. This array stores the optimal
        policies for each state. An entry `theta_0[i]` gives the optimal
        policy for state `i`. This attribute should be retrieved only after
        calling `run`.
    theta_1 : numpy.ndarray
        A float array of shape `(N + 1,)`. This array stores the optimal
        thresholds for privileged population for each state. An entry
        `theta_1[i]` gives the optimal privileged threshold for state `i`. This
        entry should be retrieved only after calling `run`.
    """


    def __init__(self,
                 dist,
                 sigma,
                 tau,
                 p_A,
                 p_D,
                 N,
                 gamma,
                 alpha,
                 discretization = 2000):
        assert callable(getattr(dist, "allowed_actions", None)) and \
               callable(getattr(dist, "theta_1_from_theta_0", None)) and \
               callable(getattr(dist, "get_payoff", None)) and \
               callable(getattr(dist, "phi_0_post", None)), \
               "The distribution does not implement all required methods"
        self.dist = dist

        assert 0 <= sigma <= 1, \
               "Ability multiplier has to be in [0, 1]"
        self.sigma = sigma

        assert 0 <= tau <= 1, \
               "Privilege multiplier has to be in [0, 1]"
        assert tau + sigma <= 1, \
               "Ability and privilege multipliers have to sum to at most 1"
        self.tau = tau

        assert 0 <= p_A <= 1 and 0 <= p_D <= 1, \
               "Transition probabilities have to be in [0, 1]"
        self.p_A = p_A
        self.p_D = p_D

        assert isinstance(N, int) and N > 0, \
               "The number of agents has to be a positive integer"
        self.N = N

        assert 0 < gamma < 1, \
               "The discount factor has to be in (0, 1)"
        self.gamma = gamma

        assert 0 <= alpha <= 1, \
               "The maximum fraction of opportunities has to be in [0, 1]"
        self.alpha = alpha

        assert isinstance(discretization, int) and discretization > 0, \
               "The discretization has to be a positive integer"
        self.discretization = discretization

        self.Q = np.zeros((N + 1, self.discretization + 1),
                          dtype = float)
        self.R = np.zeros((self.N + 1, self.discretization + 1),
                          dtype = float)
        self.S = np.zeros((self.N + 1, self.discretization + 1),
                          dtype = np.int32)
        self.mask = np.zeros((self.N + 1, self.discretization + 1),
                             dtype = np.int32)
        self.theta_0 = np.zeros(self.N + 1,
                                dtype = float)
        self.theta_1 = np.zeros(self.N + 1,
                                dtype = float)

        for s in range(self.N + 1):
            phi_0 = s / self.N
            lower, upper = dist.allowed_actions(phi_0 = phi_0,
                                                sigma = self.sigma,
                                                alpha = self.alpha)
            # Discretize allowed actions
            lower = int(lower * self.discretization/ self.sigma)
            upper = int(upper * self.discretization / self.sigma)

            self.mask[s, lower : upper + 1] = 1
            
            # Find payoffs for all allowed policies
            thetas = np.arange(lower, upper + 1) * \
                     self.sigma / self.discretization
            self.R[s, lower : upper + 1] = dist.get_payoff(theta_0 = thetas,
                                                           phi_0 = phi_0,
                                                           sigma = self.sigma,
                                                           tau = self.tau,
                                                           alpha = self.alpha)

            # Find the new state for each policy
            phi_0_posts = self.dist.phi_0_post(thetas, phi_0, self.sigma)
            # NB: This is a general formula that applies to any distribution
            phi_0_news = phi_0_posts * (1.0 - self.p_D) + \
                         (phi_0_posts ** 2) * self.p_D + \
                         (1 - phi_0_posts) * self.p_A * phi_0_posts
            # Discretize new states and update `S`
            s_news = (phi_0_news * self.N).astype(int)
            self.S[s, lower : upper + 1] = s_news

                
    def run(self, epsilon = 1e-4):
        """
        Solves for optimal policies using infinite-horizon value iteration.

        Parameters
        ----------
        epsilon : float (optional)
            The threshold for terminating the value iteration. If the `Q` matrix
            is updated by less than `epsilon` in an iteration, we consider the
            algorithm to be converged.

        Returns
        -------
        phi_0, theta_0, theta_1
            All of these are float arrays of shape `(N + 1,)`.
            `phi_0` contains all (non-discretized) states.
            `theta_0` contains the corresponding optimal policies (i.e.
            thresholds for the unprivileged population). `theta_1` contains the
            corresponding thresholds for the privileged population.
        """
        while True:
            # Perform an update, but multiply by `mask` to get rid of the
            # entries for disallowed state-policy pairs.
            Q_new = (self.R + self.gamma * self.Q.max(axis = 1)[self.S]) * \
                    self.mask
            max_e = np.max(np.abs(self.Q - Q_new))
            print("diff:", max_e)
            self.Q = Q_new
            if max_e < epsilon:
                break
        
        phi_0 = np.linspace(0, 1, self.N + 1)
        self.theta_0 = self.Q.argmax(axis=1) * self.sigma / self.discretization
        self.theta_1 = self.dist.theta_1_from_theta_0(self.theta_0,
                                                      phi_0,
                                                      self.sigma,
                                                      self.tau,
                                                      self.alpha)
        return phi_0, self.theta_0, self.theta_1

