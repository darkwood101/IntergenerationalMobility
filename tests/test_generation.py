import unittest

from aamodel.generation import generation
from aamodel.agent import privilege
from tests.helpers import helpers
from random import random

class generation_test(unittest.TestCase):
    def test_init(self):
        # `phi_0 == 0` -- everyone is privileged
        sigma, tau = helpers.sigma_tau_random()
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = sigma,
                       tau = tau,
                       phi_0 = 0,
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        self.assertEqual(len(g.agents), N)
        for a in g.agents:
            self.assertTrue(a.is_privileged())
            self.assertFalse(a.maybe_given)
            self.assertFalse(a.given)
            self.assertFalse(a.succeeded)

        # `phi_0 = 1` -- everyone is unprivileged
        sigma, tau = helpers.sigma_tau_random()
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = sigma,
                       tau = tau,
                       phi_0 = 1,
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        self.assertEqual(len(g.agents), N)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
            self.assertFalse(a.maybe_given)
            self.assertFalse(a.given)
            self.assertFalse(a.succeeded)

        # Ensure that initializing a new generation from the previous one
        # preserves the correct `phi_0`
        sigma, tau = helpers.sigma_tau_random()
        g = generation(N = helpers.N_random(),
                       sigma = sigma,
                       tau = tau,
                       phi_0 = helpers.phi_0_random(),
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            n_unprivileged = 0
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
                n_unprivileged += int(a.is_privileged())
            self.assertAlmostEqual(1 - n_unprivileged / new_g.N, new_g.phi_0)
            g = new_g


    def test_step(self):
        # `sigma = 1, a_i = 1` -- everyone succeeds
        # `p_A = p_D = 0` -- children inherit status
        g = generation(N = helpers.N_random(),
                       sigma = 1,
                       tau = 0,
                       phi_0 = helpers.phi_0_random(),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best,
                       agents = None)
        
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            self.assertAlmostEqual(payoff, 1.0)
            self.assertEqual(g.n_successes, g.N)
            self.assertAlmostEqual(new_g.phi_0, 0.0)
            for a in new_g.agents:
                self.assertTrue(a.is_privileged())
                self.assertFalse(a.maybe_given)
            g = new_g

        # `sigma = 0, tau = 0` -- everyone fails
        # `p_A = p_D = 0` -- children inherit status
        g = generation(N = helpers.N_random(),
                       sigma = 0,
                       tau = 0,
                       phi_0 = helpers.phi_0_random(),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            self.assertAlmostEqual(payoff, 0.0)
            self.assertEqual(g.n_successes, 0)
            self.assertAlmostEqual(new_g.phi_0, g.phi_0)
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
            g = new_g

        # `sigma = 1, a_i = 1` -- everyone succeeds
        # `p_A = p_D = 0` -- children inherit status
        # `phi_0 = 1` -- everyone starts out as unprivileged
        g = generation(N = helpers.N_random(),
                       sigma = 1,
                       tau = 0,
                       phi_0 = 1,
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best,
                       agents = None)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            self.assertAlmostEqual(payoff, 1.0)
            self.assertEqual(g.n_successes, new_g.N)
            self.assertAlmostEqual(new_g.phi_0, 0.0)
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
                self.assertTrue(a.is_privileged())
            for a in g.agents:
                self.assertTrue(a.maybe_given)
                self.assertTrue(a.given)
                self.assertTrue(a.is_privileged())
                self.assertTrue(a.succeeded)
            g = new_g

        # `sigma = 1, a_i = 1` -- everyone succeeds
        # `p_A = p_D = 0` -- children inherit status
        # `phi_0 = 1` -- everyone starts out as unprivileged
        g = generation(N = helpers.N_random(),
                       sigma = 1,
                       tau = 0,
                       phi_0 = 1,
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best,
                       agents = None)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            self.assertAlmostEqual(payoff, 1.0)
            self.assertEqual(g.n_successes, new_g.N)
            self.assertAlmostEqual(new_g.phi_0, 0.0)
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
                self.assertTrue(a.is_privileged())
            for a in g.agents:
                self.assertTrue(a.maybe_given)
                self.assertTrue(a.given)
                self.assertTrue(a.is_privileged())
                self.assertTrue(a.succeeded)
            g = new_g

        # `theta_0 = theta_1 = 1, sigma = tau = 0.5` -- thresholds are too high
        # so nobody will be given the opportunity
        # `p_A = p_D = 0` -- children inherit status
        g = generation(N = helpers.N_random(),
                       sigma = 0.5,
                       tau = 0.5,
                       phi_0 = helpers.phi_0_random(),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(1, 1)
            self.assertAlmostEqual(payoff, 0.0)
            self.assertEqual(g.n_successes, 0)
            self.assertAlmostEqual(new_g.phi_0, g.phi_0)
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
            for a in g.agents:
                self.assertTrue(a.maybe_given)
                self.assertFalse(a.given)
                self.assertFalse(a.succeeded)
            g = new_g

        # `theta_0 = theta_1 = 0, sigma = tau = 0` -- everyone will get the
        # opportunity, but nobody will succeed
        # `p_A = p_D = 0` -- children inherit status
        g = generation(N = helpers.N_random(),
                       sigma = 0,
                       tau = 0,
                       phi_0 = helpers.phi_0_random(),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(0, 0)
            self.assertAlmostEqual(payoff, 0.0)
            self.assertEqual(g.n_successes, 0)
            self.assertAlmostEqual(new_g.phi_0, g.phi_0)
            for a in new_g.agents:
                self.assertFalse(a.maybe_given)
            for a in g.agents:
                self.assertTrue(a.maybe_given)
                self.assertTrue(a.given)
                self.assertFalse(a.succeeded)
            g = new_g

        # `p_A = p_D = 1` -- children's privilege is always redistributed
        # `phi_0 = 1` -- all redistribution results in unprivileged children
        sigma, tau = helpers.sigma_tau_random()
        g = generation(N = helpers.N_random(),
                       sigma = sigma,
                       tau = tau,
                       phi_0 = 1,
                       p_A = 1,
                       p_D = 1,
                       a_dist = helpers.a_dist_uniform,
                       agents = None)
        
        # Check for next 10 generations
        for _ in range(10):
            new_g, payoff = g.step(random(), random())
            self.assertAlmostEqual(new_g.phi_0, 1.0)
            for a in new_g.agents:
                self.assertFalse(a.is_privileged())
                self.assertFalse(a.maybe_given)
            g = new_g
