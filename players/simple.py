import logging
from typing import List

import numpy as np

from constants import player_data
from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Simple(BasePlayer):
    caution_multiplier: float
    alert: bool

    def __init__(self, env):
        super().__init__(env)
        self.caution_multiplier = 1
        self.alert = False
        self._pW = 1

    def plan(self):
        """
        Computes the work utility minus the home (accounts for the
        probability of getting the virus) and plays the best response
        """
        assert len(self.action_plan) == 0

        surplus_risk = self.caution_multiplier * self.w_infection_risk - self.h_infection_risk
        u_v = surplus_risk * (self.death_risk * self.u_death)
        work = self.u_economic_w + u_v
        cutoff = np.sqrt(-u_v / self.u_economic_max)

        p_h_mean = 1
        p_h_delta = player_data['p-healthy-fluctuation']
        p = (p_h_mean + p_h_delta / 2 - cutoff) / p_h_delta
        p = min(1, max(0, p))
        self._pW = p
        logger.info("cutoff: {1:.2f} S prob: {0:.2f}".format(p, cutoff))
        self.action_plan.append("W" if work > 0 else "H")

        logger.debug("Estimated (self) work payoff: {0}".format(work))

    def update(self, actions: List[str], self_idx: int):
        work_people = actions.count("W")
        total_people = len(actions)
        attendance = work_people / total_people
        extra_caution = 100 if self.alert else 1
        if attendance > 0.75:
            self.caution_multiplier = 2 * attendance * attendance * extra_caution
        else:
            self.caution_multiplier = 1 * extra_caution
        self.alert = False

    def on_alert(self):
        self.alert = True

    @property
    def pW(self) -> float:
        return self._pW
