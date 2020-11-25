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

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.caution_multiplier = 1

    def plan(self):
        """
        Computes the work utility minus the home (accounts for the
        probability of getting the virus) and plays the best response
        """
        assert len(self.action_plan) == 0

        cash_work = self.u_economic_w
        virus_util = self.u_virus
        death_risk = self.death_risk
        death_util = self.u_death
        surplus_risk = self.caution_multiplier * self.w_infection_risk - self.h_infection_risk

        work = cash_work + surplus_risk * (virus_util + death_risk * death_util)
        self.action_plan.append("W" if work > 0 else "H")

        logger.debug("Estimated work payoff: {0}".format(work))

    def update(self, actions: List[str], self_idx: int):
        work_people = actions.count("W")
        total_people = len(actions)
        attendance = work_people / total_people
        if attendance > 0.5:
            self.caution_multiplier = 4 * attendance * attendance
