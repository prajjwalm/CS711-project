import logging
from typing import List

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

    def plan(self):
        """
        Computes the work utility minus the home (accounts for the
        probability of getting the virus) and plays the best response
        """
        assert len(self.action_plan) == 0

        surplus_risk = self.caution_multiplier * self.w_infection_risk - self.h_infection_risk
        work = self.u_economic_w + surplus_risk * (self.death_risk * self.u_death)
        self.action_plan.append("W" if work > 0 else "H")

        logger.debug("Estimated (self) work payoff: {0}".format(work))

    def update(self, actions: List[str], self_idx: int):
        work_people = actions.count("W")
        total_people = len(actions)
        attendance = work_people / total_people
        extra_caution = 10 if self.alert else 1
        if attendance > 0.75:
            self.caution_multiplier = 2 * attendance * attendance * extra_caution
        else:
            self.caution_multiplier = 1 * extra_caution
        self.alert = False

    def on_alert(self):
        self.alert = True
