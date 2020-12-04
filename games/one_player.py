import argparse
import logging

from enviornments import BaseEnvironment
from players import BasePlayer, Coward, Planner, Simple

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


player_types = {
    "c": Coward,
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

    def __init__(self, player: BasePlayer):
        self.p = player
        self.env = player.env

    def simulate(self):
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

                if action == "W" and self.p.t_i is None:
                    self.p.n_w += 1

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

        if self.p.t_i is not None:
            print("Went to work {0:d} days before getting infected on the {1:d}th day".format(self.p.n_w, self.p.t_i))
        else:
            print("Went to work {0:d} days, didn't get infected".format(self.p.n_w))
