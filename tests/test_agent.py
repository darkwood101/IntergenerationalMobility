import unittest
from aamodel.agent import agent, privilege
from tests.helpers import helpers
from random import random

# Sometimes, we don't care about values of certain parameters, so we set
# them to `random()`. The only current exception to this is `sigma` and
# `tau` parameters. For these, we need to force the rule
# `sigma + tau <= 1.0`, so we can't set them fully randomly.

# Tests for `agent`
class agent_test(unittest.TestCase):

    # Test if drawing from distributions works as expected
    def test_a_dist(self):
        # Ability drawn from uniform has to be in [0, 1]
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = helpers.c_random(),
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertLessEqual(0, a.a)
        self.assertLessEqual(a.a, 1)

        # Best ability has to be 1.0
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_best,
                  c = helpers.c_random(),
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertAlmostEqual(a.a, 1.0)

        # Worst ability has to be 0.0
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_worst,
                  c = helpers.c_random(),
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertAlmostEqual(a.a, 0.0)


    # Test the `is_privileged()` method without allocating opportunities
    def test_privilege(self):
        # Force `is_privileged() == True`
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.PRIVILEGED,
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertTrue(a.is_privileged())

        # Force `is_privileged() == False`
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.is_privileged())


    # Test `maybe_give_opportunity` and `is_privileged()` in combination
    def test_maybe_given(self):
        # `tau = 1` -- privileged agents guaranteed to succeed
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.PRIVILEGED,
                  sigma = 0,
                  tau = 1,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(a.is_privileged())
        self.assertTrue(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertTrue(a.given)
        self.assertTrue(a.succeeded)
        self.assertTrue(a.is_privileged())

        # `sigma = 0` -- unprivileged agents can never succeed
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = 0,
                  tau = 1,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(a.is_privileged())
        self.assertFalse(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(a.is_privileged())

        # `sigma = 1, a_i = 1` -- unprivileged agents guaranteed to succeed
        a = agent(a_dist = helpers.a_dist_best,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = 1,
                  tau = 0,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(a.is_privileged())
        self.assertTrue(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertTrue(a.given)
        self.assertTrue(a.succeeded)
        self.assertTrue(a.is_privileged())

        # `sigma = 1, a_i = 1` -- privileged agents guaranteed to succeed
        a = agent(a_dist = helpers.a_dist_best,
                  c = privilege.PRIVILEGED,
                  sigma = 1,
                  tau = 0,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(a.is_privileged())
        self.assertTrue(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertTrue(a.given)
        self.assertTrue(a.succeeded)
        self.assertTrue(a.is_privileged())

        # `a_i = 0` -- unprivileged agents can never succeed
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_worst,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = sigma,
                  tau = tau,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(a.is_privileged())
        self.assertFalse(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(a.is_privileged())

        # `a_i = 0, tau = 0` -- privileged agents can never succeed
        a = agent(a_dist = helpers.a_dist_worst,
                  c = privilege.PRIVILEGED,
                  sigma = 1,
                  tau = 0,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(a.is_privileged())
        self.assertFalse(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(a.is_privileged())

        # `a_i = 0, tau = 1` -- privileged agents guaranteed to succeed
        a = agent(a_dist = helpers.a_dist_worst,
                  c = privilege.PRIVILEGED,
                  sigma = 0,
                  tau = 1,
                  p_A = helpers.p_A_random(),
                  p_D = helpers.p_D_random())
        self.assertFalse(a.maybe_given)
        self.assertTrue(a.is_privileged())
        self.assertTrue(a.maybe_give_opportunity(random(), random()))
        self.assertTrue(a.maybe_given)
        self.assertTrue(a.given)
        self.assertTrue(a.succeeded)
        self.assertTrue(a.is_privileged())


    def test_offspring(self):
        # `p_A = p_D = 0` -- children always inherit privilege
        sigma, tau = helpers.sigma_tau_random()
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = helpers.c_random(),
                  sigma = sigma,
                  tau = tau,
                  p_A = 0,
                  p_D = 0)
        a.maybe_give_opportunity(random(), random())
        self.assertTrue(a.maybe_given)
        parent_a_c = a.c
        a.produce_offspring(phi_0)
        self.assertFalse(a.maybe_given)
        self.assertEqual(a.c, parent_a_c)

        # `p_A = p_D = 1` -- children's privilege is always redistributed
        # `phi_0 = 0` -- all redistribution results in privileged children
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = sigma,
                  tau = tau,
                  p_A = 1,
                  p_D = 1)
        self.assertFalse(a.maybe_given)
        a.maybe_give_opportunity(random(), random())
        self.assertTrue(a.maybe_given)
        a.produce_offspring(0)
        self.assertFalse(a.maybe_given)
        self.assertTrue(a.is_privileged())

        # `p_A = p_D = 1` -- children's privilege is always redistributed
        # `phi_0 = 1` -- all redistribution results in unprivileged children
        sigma, tau = helpers.sigma_tau_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = privilege.PRIVILEGED,
                  sigma = sigma,
                  tau = tau,
                  p_A = 1,
                  p_D = 1)
        self.assertFalse(a.maybe_given)
        a.maybe_give_opportunity(random(), random())
        self.assertTrue(a.maybe_given)
        a.produce_offspring(1)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.is_privileged())

    def test_step(self):
        # `sigma = 1, a_i = 1` -- everyone guaranteed to succeed
        # `p_A = 0, p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_best,
                  c = helpers.c_random(),
                  sigma = 1,
                  tau = 0,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        succeeded = a.step(random(), random(), phi_0)
        self.assertTrue(a.is_privileged())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(succeeded)

        # `sigma = 0` -- unprivileged agents can never succeed
        # `p_A = p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_best,
                  c = privilege.NOT_PRIVILEGED,
                  sigma = 0,
                  tau = 1,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        succeeded = a.step(random(), random(), phi_0)
        self.assertFalse(a.is_privileged())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(succeeded)

        # `tau = 1` -- privileged agents will always succeed
        # `p_A = p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_best,
                  c = privilege.PRIVILEGED,
                  sigma = 0,
                  tau = 1,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        succeeded = a.step(random(), random(), phi_0)
        self.assertTrue(a.is_privileged())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertTrue(succeeded)

        # `sigma = 0, tau = 0` -- nobody can succeed
        # `p_A = p_D = 1, phi_0 = 0` -- children always end up privileged
        a = agent(a_dist = helpers.a_dist_best,
                  c = helpers.c_random(),
                  sigma = 0,
                  tau = 0,
                  p_A = 1,
                  p_D = 1)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        succeeded = a.step(random(), random(), 0)
        self.assertTrue(a.is_privileged())
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(succeeded)

        # `sigma = 0, tau = 0` -- nobody can succeed
        # `p_A = p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_best,
                  c = helpers.c_random(),
                  sigma = 0,
                  tau = 0,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        parent_c = a.c
        succeeded = a.step(random(), random(), phi_0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(succeeded)
        self.assertEqual(parent_c, a.c)

        # `theta_0 = theta_1 = 1, sigma = tau = 0.5` -- thresholds are too high
        # so nobody will be given the opportunity
        # `p_A = p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = helpers.c_random(),
                  sigma = 0.5,
                  tau = 0.5,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        parent_c = a.c
        succeeded = a.step(1, 1, phi_0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(succeeded)
        self.assertEqual(parent_c, a.c)

        # `theta_0 = theta_1 = 0, sigma = tau = 0` -- everyone will get the
        # opportunity, but nobody will succeed
        # `p_A = p_D = 0` -- children will always inherit privilege
        phi_0 = helpers.phi_0_random()
        a = agent(a_dist = helpers.a_dist_uniform,
                  c = helpers.c_random(),
                  sigma = 0,
                  tau = 0,
                  p_A = 0,
                  p_D = 0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        parent_c = a.c
        succeeded = a.step(0, 0, phi_0)
        self.assertFalse(a.maybe_given)
        self.assertFalse(a.given)
        self.assertFalse(a.succeeded)
        self.assertFalse(succeeded)
        self.assertEqual(parent_c, a.c)

