import logging
from typing import List

import numpy as np

from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Planner(_Person):
    last_week_actions: List[str]
    target_days: int

    # f(no. of days worked last week)
    caution_multiplier: np.ndarray
    utility_multiplier: np.ndarray

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_week_actions = []
        self.target_days = 4
        self.caution_multiplier = np.power(2.0, np.arange(8) - self.target_days)
        self.utility_multiplier = np.power(2.0, self.target_days - np.arange(8))

        logger.debug("Caution multiplier: " + str(self.caution_multiplier))
        logger.debug("Utility multiplier: " + str(self.utility_multiplier))

    def plan(self):
        """
        Strategy:
        Sets a target working days per week, as the number of working days in
        the last 7 days falls below the target increases perceived work utility,
        the more it rises above the target, increases caution
        """

        assert len(self.action_plan) == 0

        if len(self.last_week_actions) >= 7:
            self.last_week_actions.pop(0)

        n_w = self.last_week_actions.count("W")
        u_pos = self.u_economic_w * self.utility_multiplier[n_w]
        u_neg = \
            self.w_infection_risk * self.caution_multiplier[n_w] \
            * self.death_risk * self.u_death * self.p_healthy

        action = "W" if u_pos + u_neg > 0 else "H"
        logger.debug("\n  Economic utility is: {0:.2f} [{3:.2f}x multiplier]"
                     "\n  Virus utility is {1:.2f}  [{4:.2f}x multiplier]"
                     "\n  Net = {2:.2f}, so {5}"
                     "".format(u_pos, u_neg, u_pos + u_neg,
                               self.utility_multiplier[n_w],
                               self.caution_multiplier[n_w], action))

        self.action_plan.append(action)
        self.last_week_actions.append(action)
