import logging

from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class TypeRi(_Person):

    def plan(self):
        assert len(self.action_plan) == 0

        home_days: int
        work_days: int
        cash_home = self.u_economic["H"]
        cash_work = self.u_economic["W"]
        virus_util = self.u_virus
        death_risk = self.death_risk
        surplus_risk = self.work_infection_risk
        death_util = self.u_death

        work = cash_work * (1 - surplus_risk) #+ surplus_risk * (virus_util + death_risk * death_util)
        home = cash_home
        if work >= 0 and home <= 0 :
            work_days = 7
            home_days = 0
        elif work <= 0 and home > 0 :
            work_days = 0
            home_days = 7
        else:
            work_days = (work/(work+home))*7
            home_days = 7 - work_days

        for i in range(work_days):
            self.action_plan.append("W")
        for i in range(home_days):
            self.action_plan.append("H")

        logger.debug("Economic payoffs are: {0}; so choosing {1}".format(
                work_days, home_days))

