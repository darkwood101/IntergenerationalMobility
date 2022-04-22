import numpy as np
from math import isclose

class uniform_distribution:
    """
    Contains helpful methods for the uniform distribution of abilities that the
    solver can use. Since currently we only support the [0, 1] uniform
    distribution, this class does not store any attributes.

    The common arguments to methods are as follows.

    phi_0 : float
        The unprivileged fraction of population. Always has to be in [0, 1].
    sigma : float
        The ability multiplier. Always has to be in [0, 1].
    tau : float
        The privilege multiplier. Always has to be in [0, 1]. In addition, we
        need to have `sigma + tau <= 1`.
    alpha : float
        The maximum amount of opportunities that are available to be
        allocated, expressed as a fraction of the population. Has to be in
        [0, 1].
    theta_0 : numpy.ndarray
        An array of allocation thresholds for the unprivileged population.
        Each element has to be in [0, `sigma`]. We perform batched computation
        on `theta_0` instead of computing on each element individually in order
        to improve performance.
    """


    def __init__(self, a = 0, b = 1):
        # Currently we force `a == 0` and `b == 1`, but we allow for the
        # possibility of extension.
        assert isclose(a, 0.0) and isclose(b, 1.0)


    def allowed_actions(self, phi_0, sigma, alpha):
        """
        Computes a range of actions that are permissible for a given `alpha`
        threshold. Using an action outside of this range would cause either
        less than `alpha` opportunities to be allocated, or more.

        NB: This function does not perform any error checking on its arguments.

        Parameters (explained in class docstring)
        -----------------------------------------
        phi_0 : float
        sigma : float
        alpha : float

        Returns
        -------
        [lower, upper] : Tuple[float, float]
            Range of allowed actions. We guarantee that both `lower` and `upper`
            are in [0, `sigma`], and that `lower <= upper`.
        """
        # Special case: `phi_0 == 0`, avoid division by 0.
        # Any action is allowed in this case (it won't have any effect anyway,
        # because there is no unprivileged population).
        if isclose(phi_0, 0.0):
            lower = 0
            upper = sigma
        else:
            # Cap [`lower`, `upper`] to [0, `sigma`].
            lower = max(sigma * (1.0 - alpha / phi_0), 0)
            upper = min(sigma * (1.0 - alpha) / phi_0, sigma)
        assert 0 <= lower <= upper <= sigma
        return lower, upper


    def theta_1_from_theta_0(self, theta_0, phi_0, sigma, tau, alpha):
        """
        Finds `theta_1` thresholds given `theta_0` thresholds such that the
        total amount of allocated opportunities is equal to `alpha` fraction
        of the population for any combination of thresholds.

        NB: This function does not perform any error checking on its arguments.

        Parameters (explained in class docstring)
        -----------------------------------------
        theta_0 : numpy.ndarray
        phi_0 : numpy.ndarray or scalar
            Here, we allow the possibility of batched computation over multiple
            `phi_0`.
        sigma : float
        tau : float
        alpha : float

        Returns
        -------
        theta_1 : numpy.ndarray
            An array of thresholds that correspond element-wise to `theta_0`
            thresholds.
        """
        # Suppress warnings if we divide by 0.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            theta_1 = (sigma * (1.0 - alpha) + \
                      (1.0 - phi_0) * tau - phi_0 * theta_0) / (1.0 - phi_0)
        # Special case: `phi_0 = 1`, fix division by 0 above.
        # In this case, `theta_1` can be anything in [`tau`, `tau + sigma`],
        # as it doesn't have any effect (there is no privileged population).
        # We choose `tau + sigma` arbitrarily.
        theta_1 = np.where(np.isclose(phi_0, 1.0), tau + sigma, theta_1)
        return theta_1


    def get_payoff(self, theta_0, phi_0, sigma, tau, alpha):
        """
        Computes payoffs for given thresholds `theta_0`. The payoff is defined
        as the fraction of population that succeeds.

        This function will throw an assertion error if any element of `theta_0`
        exceeds `sigma`.

        Parameters (explained in class docstring)
        -----------------------------------------
        theta_0 : numpy.ndarray
        phi_0 : float
        sigma : float
        tau : float
        alpha : float

        Returns
        -------
        payoffs : numpy.ndarray
            An array of payoffs, where each element corresponds to an element of
            `theta_0`. We guarantee that each element of `payoffs` is in
            [0, `alpha`]. Having a payoff greater than `alpha` would mean that
            more than `alpha` opportunities were allocated.
        """
        assert np.all(theta_0 <= sigma)

        theta_1 = self.theta_1_from_theta_0(theta_0, phi_0, sigma, tau, alpha)
        # Compute possible payoffs for privileged population. The reason we say
        # "possible" is because these payoffs won't work when
        # `tau + sigma < theta_1`. This is why we suppress warnings.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            possible_priv_payoffs = ((1.0 - phi_0) / (2 * sigma)) * \
                                    ((sigma + tau) ** 2 - theta_1 ** 2)

        # Everywhere where `tau + sigma < theta_1`, replace payoff with 0
        # (nobody from privileged population gets an opportunity).
        priv_payoffs = np.where(tau + sigma < theta_1, 0, possible_priv_payoffs)

        # Compute payoffs for unprivileged population
        unpriv_payoffs = (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)

        payoffs = unpriv_payoffs + priv_payoffs
        assert np.all(0 <= payoffs) and np.all(payoffs <= alpha)
        return payoffs


    def phi_0_post(self, theta_0, phi_0, sigma):
        """
        Computes the new fractions of unprivileged population _after_ the
        opportunities are allocated. The new fractions are computed for each
        threshold in `theta_0`.

        Parameters (explained in class docstring)
        -----------------------------------------
        theta_0 : numpy.ndarray
        phi_0 : float
        sigma : float

        Returns
        -------
        phi_0_post : numpy.ndarray
            An array of states (i.e. fractions of unprivileged population)
            after opportunity allocation that corresponds element-wise to
            `theta_0`.
        """
        assert np.all(theta_0 <= sigma)
        return phi_0 - (phi_0 / (2 * sigma)) * (sigma ** 2 - theta_0 ** 2)

