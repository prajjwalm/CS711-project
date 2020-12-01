import logging
from typing import List

import numpy as np

from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Planner(BasePlayer):
    last_week_actions: List[str]
    target_days: int
    target_ratio: float
    h: float = 0.5

    # f(no. of days worked last week)
    caution_multiplier: np.ndarray
    utility_multiplier: np.ndarray

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_week_actions = []
        # self.target_days = 4
        # self.caution_multiplier = np.power(2.0, np.arange(8) - self.target_days)
        # self.utility_multiplier = np.power(2.0, self.target_days - np.arange(8))

        """
        instead of giving equal weightage of work to last 7 days
        person gives geometrically reducing weightage to previous days
        So 1 day ago has impact of 1, 2 days ago 1/h, 3 days ago 1/h^2
        n days ago 1/h^(n-1)
        """

        self.target_ratio = 1.13
        self.caution_multiplier = lambda x: np.power(2.0, self.target_ratio / x)
        self.utility_multiplier = lambda x: np.power(2.0, x / self.target_ratio)

        self.work_weightage = np.logspace(0, 6, num=7, base=self.h)

        logger.debug("Caution multiplier: " + str(self.caution_multiplier))
        logger.debug("Utility multiplier: " + str(self.utility_multiplier))

        self.alert = False

    def plan(self):
        """
        Strategy:
        Sets a target working days per week, as the number of working days in
        the last 7 days falls below the target increases perceived work utility,
        the more it rises above the target, increases caution


        work measure = \sum_i=t to 0 [w_i if worked at i else 0] (@t+1)
        previously,
            w_i = 1 if i in last 7 days else 0

        if last_week_actions = [WWHHWWH] # something random
        now,
            w_i = 1 - (t - i)/7
        """

        assert len(self.action_plan) == 0

        if len(self.last_week_actions) >= 7:
            self.last_week_actions.pop(0)

        # I got what you're saying
        # change kar sakte hai
        # but will it help in population.py? lets sequence

        # n_w = self.last_week_actions.count("W")

        work_last_week = (np.array(self.last_week_actions) == "W") * 1.0
        r_w = np.sum(np.multiply(work_last_week, self.work_weightage))

        u_pos = self.u_economic_w * self.utility_multiplier(r_w)
        u_neg = \
            self.w_infection_risk * self.caution_multiplier(r_w) \
            * self.death_risk * self.u_death * self.p_healthy

        action = "W" if u_pos + u_neg > 0 else "H"
        logger.debug("\n  Economic utility is: {0:.2f} [{3:.2f}x multiplier]"
                     "\n  Virus utility is {1:.2f}  [{4:.2f}x multiplier]"
                     "\n  Net = {2:.2f}, so {5}"
                     "".format(u_pos, u_neg, u_pos + u_neg,
                               self.utility_multiplier(r_w),
                               self.caution_multiplier(r_w), action))

        self.action_plan.append(action)
        self.last_week_actions.append(action)

    def update(self, actions: List[str], self_idx: int):
        working_people = actions.count("W")
        total_people = len(actions)

        # If less people go to work, player feels inclined to go cause low risk
        # but less people means less payoff
        # there is some expected population that he thinks should be in office

        p = working_people / total_people
        expected_p = 0.6  # arbitrarily taken

        # come up with better method to implement
        self.caution_multiplier = lambda x: np.power(2.0, self.target_ratio / x) * p / expected_p
        self.utility_multiplier = lambda x: np.power(2.0, x / self.target_ratio) * expected_p / p

        self.caution_multiplier *= 10 if self.alert else 1

        self.alert = False

    def on_alert(self):
        self.alert = True
