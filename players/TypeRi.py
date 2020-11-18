import logging
import numpy as np
from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class TypeRi(_Person):

    def plan(self):
        if len(self.action_plan) != 0:
            return

        #home_days: int
        work_days: int
        cash_home = self.u_economic["H"]
        cash_work = self.u_economic["W"]
        # virus_util = self.u_virus
        # death_risk = self.death_risk
        # surplus_risk = self.work_infection_risk
        # death_util = self.u_death

        # + surplus_risk * (virus_util + death_risk * death_util)
        # if cash_work >= 0 and cash_home <= 0:
        #     work_days = 7
        #     home_days = 0
        # elif cash_work <= 0 and cash_home > 0:
        #     work_days = 0
        #     home_days = 7
        # else:
        #     work_days = (cash_work / (cash_work + cash_home)) * 7
        #     home_days = 7 - work_days

        virus_contact_prob_h = 1 - np.power(1 - self.home_infection_risk, np.arange(8))
        logger.debug("virus contact prob_h = " + str(virus_contact_prob_h))
        virus_contact_prob_w = 1 - np.power(1 - (self.home_infection_risk + self.work_infection_risk), np.arange(8))
        logger.debug("virus contact prob_w = " + str(virus_contact_prob_w))
        virus_contact_prob_delta = virus_contact_prob_w - virus_contact_prob_h
        logger.debug("virus contact prob_delta = " + str(virus_contact_prob_delta))
        virus_util = virus_contact_prob_delta * self.death_risk * self.u_death
        logger.debug("virus util = " + str(virus_util))
        economic_util = (cash_work - cash_home) * np.arange(8)
        logger.debug("economic util = " + str(economic_util))
        net_util = virus_util + economic_util
        logger.debug(("net util = " + str(net_util)))
        work_days = int(np.argmax(net_util))
        logger.debug("work days = " + str(work_days))

        for i in range(work_days):
            self.action_plan.append("W")
        for i in range(7 - work_days):
            self.action_plan.append("H")

