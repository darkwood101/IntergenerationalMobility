# Plot optimal `theta_0` vs `phi_0` for parameters from Figure 3 and
# `gamma = 0.8`, for both uniform and normal.
# Also plot affirmative action (like Fig 3b).

import sys
sys.path.append("..")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os.path

plt.rc('text', usetex = True)
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('figure', figsize = (5, 5))

from aamodel.solver import mdp_solver
from aamodel.uniform_distribution import uniform_distribution
from aamodel.normal_distribution import normal_distribution


def main():
    if not os.path.exists("data/"):
        os.makedirs("data/")
    if not os.path.exists("plots/"):
        os.makedirs("plots/")

    filename = sys.argv[0][:-3]

    uniform_filename = "data/" + filename + "_uniform" + ".csv"
    if not os.path.exists(uniform_filename):
        s_u = mdp_solver(dist = uniform_distribution(0, 1),
                         sigma = 0.4,
                         tau = 0.1,
                         p_A = 0,
                         p_D = 0,
                         N = 2000,
                         gamma = 0.8,
                         alpha = 0.15)
        states_u, theta_0_u, theta_1_u = s_u.run()

        np.savetxt(uniform_filename,
                   np.column_stack((states_u, theta_0_u, theta_1_u)),
                   delimiter = ",",
                   header = "states_u,theta_0_u,theta_1_u",
                   comments = "")
    else:
        df = pd.read_csv(uniform_filename)
        states_u = df["states_u"].to_numpy()
        theta_0_u = df["theta_0_u"].to_numpy()
        theta_1_u = df["theta_1_u"].to_numpy()

    normal_filename = "data/" + filename + "_normal" + ".csv"
    if not os.path.exists(normal_filename):
        s_n = mdp_solver(dist = normal_distribution(0.5, 0.05),
                         sigma = 0.4,
                         tau = 0.1,
                         p_A = 0,
                         p_D = 0,
                         N = 2000,
                         gamma = 0.8,
                         alpha = 0.15)
        states_n, theta_0_n, theta_1_n = s_n.run()

        np.savetxt(normal_filename,
                   np.column_stack((states_n, theta_0_n, theta_1_n)),
                   delimiter = ",",
                   header = "states_n,theta_0_n,theta_1_n",
                   comments = "")
    else:
        df = pd.read_csv(normal_filename)
        states_n = df["states_n"].to_numpy()
        theta_0_n = df["theta_0_n"].to_numpy()
        theta_1_n = df["theta_1_n"].to_numpy()


    plot_filename = "plots/" + filename + "_theta_0_vs_phi_0" + ".pdf"
    if not os.path.exists(plot_filename):
        plt.plot(states_u[1:], theta_0_u[1:], label = "Uniform")
        plt.plot(states_n[1:], theta_0_n[1:], label = "Normal")
        plt.title(r"$\alpha = 0.15$, $\sigma = 0.4$, $\tau = 0.1$, " \
                   "$\gamma = 0.8$")
        plt.xlabel(r"$\phi_0$")
        plt.ylabel(r"Optimal $\theta_0$")
        plt.legend(loc = "upper right")
        plt.savefig(plot_filename)
        plt.close()

    plot_filename = "plots/" + filename + "_theta_0_minus_theta_1" + ".pdf"
    if not os.path.exists(plot_filename):
        policy_diff_u = np.minimum(theta_1_u - theta_0_u, 0.4 - theta_0_u)
        policy_diff_n = np.minimum(theta_1_n - theta_0_n, 0.4 - theta_0_n)
        plt.plot(states_u[1:], policy_diff_u[1:], label = "Uniform")
        plt.plot(states_n[1:], policy_diff_n[1:], label = "Normal")
        plt.title(r"$\alpha = 0.15$, $\sigma = 0.4$, $\tau = 0.1$, " \
                   "$\gamma = 0.8$")
        plt.xlabel(r"$\phi_0$")
        plt.ylabel(r"$\min ( \sigma, \theta_1 ) - \theta_0$")
        plt.legend(loc = "upper right")
        plt.savefig(plot_filename)
        plt.close()
        

if __name__ == "__main__":
    main()
