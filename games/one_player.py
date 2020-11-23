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

                # handle utilities
                action = self.p.action_plan.pop()
                self.p.net_utility += \
                    self.p.u_economic_w if action == 'W' else 0 + self.p.u_virus

                # handle risk
                risk = \
                    self.p.w_infection_risk if action == "W" else \
                        self.p.h_infection_risk

                self.p.state_change(risk)

                if action == "W" and self.p.t_i is None:
                    self.p.n_w += 1

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
