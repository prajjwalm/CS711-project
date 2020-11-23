import logging

from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Simple(BasePlayer):

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
        surplus_risk = self.w_infection_risk - self.h_infection_risk

        work = cash_work + surplus_risk * (virus_util + death_risk * death_util)
        self.action_plan.append("W" if work > 0 else "H")

        logger.debug("Estimated work payoff: {0}".format(work))
