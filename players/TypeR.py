import logging

from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class TypeR(_Person):

    def plan(self):
        assert len(self.action_plan) == 0

        cash_home = self.u_economic["H"]
        cash_work = self.u_economic["W"]
        virus_util = self.u_virus
        death_risk = self.death_risk
        surplus_risk = self.work_infection_risk
        death_util = self.u_death
        work = cash_work*(1 - surplus_risk) + surplus_risk*(virus_util + death_risk*death_util)
        home = cash_home
        if work > home:
            self.action_plan.append("W")
        else:
            self.action_plan.append("H")

        # self.action_plan.append(max(cash, key=cash.get))

        logger.debug("Economic payoffs are: {0}, {1}; so choosing {2}".format(
                work, home, self.action_plan[-1]
        ))
