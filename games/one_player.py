import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constants import player_data, T
from enviornments import BaseEnvironment, EnvSir
from players import BasePlayer, Gullible, Planner, Simple

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


player_types = {
    "c": Gullible,
    "p": Planner,
    "s": Simple,
}


def _add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--type", choices=list(player_types.keys()), default="s",
                        help="Which player archetype to run in case of a one player game")


def _parse_args(args: argparse.Namespace, env):
    return OnePlayer(player_types[args.type](env))


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


class OnePlayer:
    p: BasePlayer
    env: BaseEnvironment

    t_utility_ot: List[float]  # total utility over time
    d_utility_ot: List[float]
    infection_rate: List[float]
    actions: List[float]
    timeline: np.ndarray

    def __init__(self, player: BasePlayer):
        self.p = player
        self.env = player.env
        self.t_utility_ot = []
        self.d_utility_ot = []
        self.infection_rate = []
        self.timeline = np.arange(self.env.max_t)
        self.actions = []

    def simulate(self):
        dead = False
        try:
            for day in self.env:
                self.p.plan()

                action = self.p.action_plan.pop()

                # handle utilities and risk
                self.actions.append(self.p.pW)
                if action == "W":
                    self.p.net_utility += self.p.u_economic_w
                    risk = self.p.w_infection_risk
                else:
                    risk = self.p.h_infection_risk

                self.p.state_change(risk)
                self.t_utility_ot.append(self.p.net_utility)
                if len(self.d_utility_ot) == 0:
                    self.d_utility_ot.append(self.p.net_utility)
                else:
                    self.d_utility_ot.append(self.t_utility_ot[-1] - self.t_utility_ot[-2])
                if action == "W" and self.p.t_i is None:
                    self.p.n_w += 1
                self.infection_rate.append(self.env.i / self.env.n)

                logger.info(
                    "True state: {0}, believes himself to be {1:d}% healthy, and has a net utility of {2:.2f}, "
                    "(percentage infected = {3:.2f}%, work risk = {4:.3f}%, home risk = {5:.3f}%)".format(
                        self.p.state,
                        int(self.p.p_healthy * 100),
                        self.p.net_utility,
                        self.env.i / self.env.n * 100,
                        self.p.w_infection_risk * 100,
                        self.p.h_infection_risk * 100
                    )
                )
        except BasePlayer.DeathException:
            logger.critical("Player Dead")
            dead = True
            self.t_utility_ot += [self.p.net_utility - player_data['u-death']] * (
                    self.env.max_t - len(self.t_utility_ot))

        if self.p.t_i is not None:
            print("Went to work {0:d} days before getting infected on the {1:d}th day".format(self.p.n_w, self.p.t_i))
            if dead:
                print("Also, he died")
        else:
            print("Went to work {0:d} days, didn't get infected".format(self.p.n_w))

        print("U", self.p.net_utility)
        return self

    def plot_graphs(self):
        def total_utility_plot():
            self.d_utility_ot = np.asarray(self.d_utility_ot)

            fig, ax1 = plt.subplots()
            color = 'tab:red'
            ax1.plot(self.timeline, self.infection_rate, color=color, label="Infection")
            ax1.set_ylabel("Infected Population")

            ax2 = ax1.twinx()
            ax2.plot(self.timeline, self.d_utility_ot)
            ax2.set_ylim(0, 2)
            ax2.set_xlabel("Time(days)")
            ax2.set_ylabel("Daily Utility")
            ax2.grid()
            ax1.legend(loc="lower right")
            fig.tight_layout()
            plt.savefig("graphs/one_player_daily_utility.jpg")

        total_utility_plot()


def main():
    t = 200
    smooth = 7
    g1 = OnePlayer(Simple(EnvSir(10000, 10, t))).simulate()
    u_s = np.zeros(t)
    g1a = np.asarray(g1.actions)
    u_s[:smooth - 1] = g1a[:smooth - 1]
    u_s[smooth - 1:] = moving_average(g1a, smooth)
    T[0] = 0
    g2 = OnePlayer(Planner(EnvSir(10000, 10, t))).simulate()
    u_p = np.zeros(t)
    g2a = np.asarray(g2.actions)
    u_p[:smooth - 1] = g2a[:smooth - 1]
    u_p[smooth - 1:] = moving_average(g2a, smooth)
    i_r = g2.infection_rate
    timeline = np.arange(len(i_r))

    fig, ax1 = plt.subplots()
    ax1.plot(timeline, i_r, color='tab:red', label="Infection")
    ax1.set_ylabel("Infected Population")

    ax2 = ax1.twinx()
    ax2.plot(timeline, u_s, color='tab:blue', label="P[W; Simple]")
    ax2.plot(timeline, u_p, color='tab:green', label="P[W; Planner]")
    ax2.set_ylim(0, 1.5)
    ax2.set_xlabel("Time(days)")
    ax2.set_ylabel("Probability of choosing Work")
    ax2.grid()
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
    # plt.savefig("graphs/one_player_daily_utility.jpg")
