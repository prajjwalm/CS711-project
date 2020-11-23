import logging

from enviornment import BaseEnvironment
from players import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class OnePlayerGame:
    p: BasePlayer
    env: BaseEnvironment

    def __init__(self, player: BasePlayer, env: BaseEnvironment):
        self.p = player
        self.env = env

    def play(self, n_days=None):
        try:
            for day in self.env:
                self.p.plan()
                self.p.act()
                logger.info(
                        "True state: {0}, believes himself to be {1:d}% "
                        "healthy, and has a net utility of {2:.2f}, "
                        "(percentage infected = {3:.2f}%, work risk = {4:.3f}%,"
                        " home risk = {5:.3f}%)".format(
                                self.p.state,
                                int(self.p.p_healthy * 100),
                                self.p.net_utility,
                                self.env.i / self.env.n * 100,
                                self.p.w_infection_risk * 100,
                                self.p.h_infection_risk * 100
                        )
                )
                if n_days is not None and day == n_days:
                    break
        except NotImplementedError:
            logger.critical("Old doc dead")

        if self.p.t_i is not None:
            print("Went to work {0:d} days before getting infected on the "
                  "{1:d}th day".format(self.p.n_w, self.p.t_i))
        else:
            print("Went to work {0:d} days, didn't get infected"
                  "".format(self.p.n_w))
