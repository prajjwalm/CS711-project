import argparse
import logging
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from constants import player_data
from enviornments import BaseEnvironment
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


class OnePlayer:
    p: BasePlayer
    env: BaseEnvironment

    t_utility_ot: List[float]  # total utility over time
    d_utility_ot: List[float]
    infection_rate: List[float]
    timeline: np.ndarray

    def __init__(self, player: BasePlayer):
        self.p = player
        self.env = player.env
        self.t_utility_ot = []
        self.d_utility_ot = []
        self.infection_rate = []
        self.timeline = np.arange(self.env.max_t)

    def simulate(self):
        dead = False
        try:
            for day in self.env:
                self.p.plan()

                action = self.p.action_plan.pop()

                # handle utilities and risk
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
