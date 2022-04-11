import unittest

from aamodel.generation import generation
from aamodel.agent import privilege
from tests.helpers import helpers
from random import random

class generation_test(unittest.TestCase):
    def test_init(self):
        # `n_privileged = N` -- everyone is privileged
        sigma, tau = helpers.sigma_tau_random()
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = sigma,
                       tau = tau,
                       n_privileged = N,
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform)
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
                       n_privileged = 0,
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform)
        self.assertEqual(len(g.agents), N)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
            self.assertFalse(a.maybe_given)
            self.assertFalse(a.given)
            self.assertFalse(a.succeeded)

        # Ensure that initializing a new generation from the previous one
        # preserves the correct `phi_0`
        sigma, tau = helpers.sigma_tau_random()
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = sigma,
                       tau = tau,
                       n_privileged = helpers.n_privileged_random(N),
                       p_A = helpers.p_A_random(),
                       p_D = helpers.p_D_random(),
                       a_dist = helpers.a_dist_uniform)
        
        # Check for next 10 generations
        for _ in range(10):
            n_successes = g.step(random(), random())
            n_privileged = 0
            for a in g.agents:
                self.assertFalse(a.maybe_given)
                self.assertFalse(a.given)
                self.assertFalse(a.succeeded)
                n_privileged += int(a.is_privileged())
            self.assertAlmostEqual(1 - n_privileged / g.N, g.phi_0)
            self.assertEqual(n_privileged, g.n_privileged)


    def test_step(self):
        # `sigma = 1, a_i = 1` -- everyone succeeds
        # `p_A = p_D = 0` -- children inherit status
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 1,
                       tau = 0,
                       n_privileged = helpers.n_privileged_random(N),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best)
        
        # Check for next 10 generations
        for _ in range(10):
            n_successes = g.step(random(), random())
            self.assertEqual(n_successes, N)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(g.phi_0, 0.0)
            for a in g.agents:
                self.assertTrue(a.is_privileged())
                self.assertFalse(a.maybe_given)

        # `sigma = 0, tau = 0` -- everyone fails
        # `p_A = p_D = 0` -- children inherit status
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 0,
                       tau = 0,
                       n_privileged = helpers.n_privileged_random(N),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform)
        
        # Check for next 10 generations
        for _ in range(10):
            old_phi_0 = g.phi_0
            n_successes = g.step(random(), random())
            self.assertEqual(n_successes, 0)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(old_phi_0, g.phi_0)
            for a in g.agents:
                self.assertFalse(a.maybe_given)

        # `sigma = 1, a_i = 1` -- everyone succeeds
        # `p_A = p_D = 0` -- children inherit status
        # `n_privileged = 0` -- everyone starts out as unprivileged
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 1,
                       tau = 0,
                       n_privileged = 0,
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
        self.assertAlmostEqual(g.phi_0, 1.0)
        # Check for next 10 generations
        for _ in range(10):
            n_successes = g.step(random(), random())
            self.assertEqual(n_successes, N)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(g.phi_0, 0.0)
            for a in g.agents:
                self.assertFalse(a.maybe_given)
                self.assertFalse(a.given)
                self.assertTrue(a.is_privileged())
                self.assertFalse(a.succeeded)

        # `sigma = 0, a_i = 0` -- everyone fails
        # `p_A = p_D = 0` -- children inherit status
        # `phi_0 = 1` -- everyone starts out as unprivileged
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 0,
                       tau = 0,
                       n_privileged = 0,
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_best)
        for a in g.agents:
            self.assertFalse(a.is_privileged())
        # Check for next 10 generations
        for _ in range(10):
            n_successes = g.step(random(), random())
            self.assertEqual(n_successes, 0)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(g.phi_0, 1.0)
            for a in g.agents:
                self.assertFalse(a.maybe_given)
                self.assertFalse(a.given)
                self.assertFalse(a.is_privileged())
                self.assertFalse(a.succeeded)

        # `theta_0 = theta_1 = 1, sigma = tau = 0.5` -- thresholds are too high
        # so nobody will be given the opportunity
        # `p_A = p_D = 0` -- children inherit status
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 0.5,
                       tau = 0.5,
                       n_privileged = helpers.n_privileged_random(N),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform)
        # Check for next 10 generations
        for _ in range(10):
            old_phi_0 = g.phi_0
            n_successes = g.step(1, 1)
            self.assertEqual(n_successes, 0)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(old_phi_0, g.phi_0)
            for a in g.agents:
                self.assertFalse(a.maybe_given)
                self.assertFalse(a.given)
                self.assertFalse(a.succeeded)

        # `theta_0 = theta_1 = 0, sigma = tau = 0` -- everyone will get the
        # opportunity, but nobody will succeed
        # `p_A = p_D = 0` -- children inherit status
        N = helpers.N_random()
        g = generation(N = N,
                       sigma = 0,
                       tau = 0,
                       n_privileged = helpers.n_privileged_random(N),
                       p_A = 0,
                       p_D = 0,
                       a_dist = helpers.a_dist_uniform)
        # Check for next 10 generations
        for _ in range(10):
            old_phi_0 = g.phi_0
            n_successes = g.step(0, 0)
            self.assertEqual(n_successes, 0)
            self.assertEqual(N, g.N)
            self.assertAlmostEqual(old_phi_0, g.phi_0)
            for a in g.agents:
                self.assertFalse(a.maybe_given)
                self.assertFalse(a.given)
                self.assertFalse(a.succeeded)

        # `p_A = p_D = 1` -- children's privilege is always redistributed
        # `phi_0 = 1` -- all redistribution results in unprivileged children
        sigma, tau = helpers.sigma_tau_random()
        N = helpers.N_random()
        g = generation(N = helpers.N_random(),
                       sigma = sigma,
                       tau = tau,
                       n_privileged = 0,
                       p_A = 1,
                       p_D = 1,
                       a_dist = helpers.a_dist_uniform)
        
        # Check for next 10 generations
        for _ in range(10):
            n_successes = g.step(random(), random())
            self.assertAlmostEqual(g.phi_0, 1.0)
            for a in g.agents:
                self.assertFalse(a.is_privileged())
                self.assertFalse(a.maybe_given)

