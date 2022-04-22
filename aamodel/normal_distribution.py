import numpy as np
from math import isclose
from scipy.stats import norm

class normal_distribution:
    """
    Contains helpful methods for the normal distribution of abilities that the
    solver can use.

    Attributes
    ----------
    mu : float
        The mean of the distribution. Currently, only 0.5 is supported.
    sd : float
        Standard deviation of the distribution.

    The methods and arguments to methods are the same as for
    `uniform_distribution`, so the descriptions are omitted. See
    `uniform_distribution` for more details.
    """


    def __init__(self, mu, sd):
        assert isclose(mu, 0.5)
        self.mu = 0.5
        self.sd = sd


    def CDF(self, x):
        return norm.cdf(x, loc = self.mu, scale = self.sd)


    # NB: `CDF_inv` stands for "CDF inverse".
    def CDF_inv(self, x):
        return norm.ppf(x, loc = self.mu, scale = self.sd)


    def PDF(self, x):
        return norm.pdf(x, loc = self.mu, scale = self.sd)


    def allowed_actions(self, phi_0, sigma, alpha):
        # Special case: `phi_0 == 0`, avoid division by 0.
        # Any action is allowed in this case (it won't have any effect anyway,
        # because there is no unprivileged population).
        if isclose(phi_0, 0.0):
            lower = 0
            upper = sigma
        else:
            # Cap [`lower`, `upper`] to [0, `sigma`].
            # Special care needs to be taken not to pass anything outside [0, 1]
            # to `CDF_inv`.
            if 1.0 - alpha / phi_0 < 0:
                lower = 0
            else:
                lower = max(sigma * self.CDF_inv(1.0 - alpha / phi_0), 0)
            if (1.0 - alpha) / phi_0 > 1:
                upper = sigma
            else:
                upper = min(sigma * self.CDF_inv((1.0 - alpha) / phi_0), sigma)
        assert 0 <= lower <= upper <= sigma
        return lower, upper


    def theta_1_from_theta_0(self, theta_0, phi_0, sigma, tau, alpha):
        # Suppress warnings if we divide by 0.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            cdf_inv_arg = (1.0 - alpha - phi_0 * self.CDF(theta_0 / sigma)) / \
                          (1.0 - phi_0)
        # This is a little hacky.
        # Even though we take all precautions to ensure that the the argument to
        # the inverse CDF is always well-defined, it can still happen that this
        # is not the case. This is due to rounding errors. These functions are
        # very sensitive to small variations, so even a tiny rounding error
        # causes a big explosion which drives the CDF inverse argument below 0
        # or above 1. This is why we just cap it here to [0, 1]. It's not the
        # most elegant solution ever, but it works.
        cdf_inv_arg = np.where(cdf_inv_arg < 0, 0, cdf_inv_arg)
        cdf_inv_arg = np.where(cdf_inv_arg > 1, 1, cdf_inv_arg)
        theta_1 = sigma * self.CDF_inv(cdf_inv_arg) + tau

        # Special case: `phi_0 = 1`, fix division by 0 above.
        # In this case, `theta_1` can be anything in [`tau`, `tau + sigma`],
        # as it doesn't have any effect (there is no privileged population).
        # We choose `tau + sigma` arbitrarily.
        theta_1 = np.where(np.isclose(phi_0, 1.0), tau + sigma, theta_1)
        assert not np.any(np.isnan(theta_1))
        return theta_1


    def get_payoff(self, theta_0, phi_0, sigma, tau, alpha):
        assert np.all(sigma >= theta_0)

        theta_1 = self.theta_1_from_theta_0(theta_0, phi_0, sigma, tau, alpha)
        # Compute possible payoffs for privileged population. The reason we say
        # "possible" is because these payoffs won't work when
        # `tau + sigma < theta_1`. This is why we suppress warnings.
        with np.errstate(divide = 'ignore', invalid = 'ignore'):
            # It's a long expression...
            # If someone knows how to break it up more nicely, be my guest
            arg = (theta_1 - tau) / sigma
            possible_priv_payoffs = (1.0 - phi_0) * (1.0 - self.CDF(arg)) * \
                                    (sigma * (self.mu + (self.sd ** 2) * \
                                    self.PDF(arg) / (1.0 - self.CDF(arg))) + \
                                    tau)
        
        # Everywhere where `tau + sigma < theta_1`, replace payoff with 0
        # (nobody from privileged population gets an opportunity).
        priv_payoffs = np.where(tau + sigma < theta_1, 0, possible_priv_payoffs)

        # Compute payoffs for unprivileged population
        arg = theta_0 / sigma
        unpriv_payoffs = phi_0 * sigma * (self.mu * (1 - self.CDF(arg)) + \
                         (self.sd ** 2) * self.PDF(arg))

        payoffs = unpriv_payoffs + priv_payoffs
        assert np.all(0 <= payoffs) and np.all(payoffs <= alpha)
        return payoffs
                         

    def phi_0_post(self, theta_0, phi_0, sigma):
        assert np.all(theta_0 <= sigma)
        arg = theta_0 / sigma
        return phi_0 - phi_0 * sigma * \
                       (self.mu * (1 - self.CDF(arg)) + \
                       (self.sd ** 2) * self.PDF(arg))

