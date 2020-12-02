import logging
from typing import List

import numpy as np

from constants import env_params
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
    h: float
    p: float

    work_weightage: np.ndarray

    #
    # P[W_t = 1] = m(i-stage, delta-i_t).X_t + c(i-stage, delta-i_t)
    # where,
    #   i-stage: index infection stage in {0, 1, ..., t-removal}
    #   delta-i: difference in infection probabilities in case of home and work
    #
    # we must have: [c + m/(1-h), c] \subset [0, 1],
    # and preferably, @X_t = 1 / 2(1-h), P[W_t = 1] = P_Simple
    #

    @staticmethod
    def m(i_stage: float, delta_i: float, c: float):
        tr = env_params['t-removal']
        ts = env_params['t-symptoms']
        return -(delta_i * np.where(i_stage != tr, np.clip(1 - i_stage / ts, 0, 1), 0)) * c

    @staticmethod
    def c(i_stage: float, delta_i: float):
        return 1 - delta_i

    # BUGGY CODE
    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)

        # shouldn't be needed anymore
        self.last_week_actions = []

        # instead of giving equal weightage of work to last 7 days
        # person gives geometrically reducing weightage to previous days
        # So 1 day ago has impact of 1, 2 days ago 1/h, 3 days ago 1/h^2
        # n days ago 1/h^(n-1)

        self.target_ratio = 1.13

        self.work_weightage = np.logspace(0, 6, num=7, base=self.h)

        self.alert = False

    def plan(self):
        """
        Strategy:

        """

        assert len(self.action_plan) == 0

        if len(self.last_week_actions) >= 7:
            self.last_week_actions.pop(0)

        coeff_i = 0.05
        coeff_wi = 1 * self.job_risk
        p_h_max = 23 / 24
        total_infectious = 0
        working_infectious = 0
        eta_ih = total_infectious * coeff_i
        eta_iw = np.clip(1 - (1 - total_infectious * coeff_i) * (1 - working_infectious * coeff_wi), eta_ih, 1)

        self.h = 1 - 1 / (2 * (
                (p_h_max - np.sqrt(self.u_death * self.death_risk * (eta_iw - eta_ih) / self.u_economic_w)) / 0.25))

        work_last_week = (np.array(self.last_week_actions) == "W") * 1.0
        work_measure = np.sum(np.multiply(work_last_week, self.work_weightage))

        self.p = 1 - (1 - self.h) * work_measure
        #        u_pos = self.u_economic_w * self.utility_multiplier(r_w)
        #        u_neg = \
        #            self.w_infection_risk * self.caution_multiplier(r_w) \
        #            * self.death_risk * self.u_death * self.p_healthy

        action = "W" if np.random.rand() < self.p else "H"
        #        logger.debug("\n  Economic utility is: {0:.2f} [{3:.2f}x multiplier]"
        #                     "\n  Virus utility is {1:.2f}  [{4:.2f}x multiplier]"
        #                     "\n  Net = {2:.2f}, so {5}"
        #                     "".format(u_pos, u_neg, u_pos + u_neg,
        #                               self.utility_multiplier(r_w),
        #                               self.caution_multiplier(r_w), action))

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

        self.alert = False

    def on_alert(self):
        self.alert = True
