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

        plan_days = 15

        virus_contact_prob_h = 1 - np.power(1 - self.home_infection_risk,
                                            np.arange(plan_days))
        logger.debug("virus contact prob_h = " + str(virus_contact_prob_h))
        virus_contact_prob_w = 1 - np.power(
            1 - (self.home_infection_risk + self.work_infection_risk),
            np.arange(plan_days))
        logger.debug("virus contact prob_w = " + str(virus_contact_prob_w))
        virus_contact_prob_delta = virus_contact_prob_w - virus_contact_prob_h
        logger.debug("virus contact prob_delta = " + str(virus_contact_prob_delta))
        virus_util = virus_contact_prob_delta * self.death_risk * self.u_death
        logger.debug("virus util = " + str(virus_util))
        economic_util = self.u_economic_w * np.arange(plan_days)
        logger.debug("economic util = " + str(economic_util))
        net_util = virus_util + economic_util
        logger.debug(("net util = " + str(net_util)))
        work_days = int(np.argmax(net_util))
        logger.debug("work days = " + str(work_days))

        for i in range(work_days):
            self.action_plan.append("W")
        for i in range(plan_days - work_days):
            self.action_plan.append("H")
