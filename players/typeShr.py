import logging
from typing import List

from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class typeShr(_Person):
    last_week_actions: List[str]

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_week_actions = []

    def plan(self):
        if len(self.last_week_actions) >= 7:
            self.last_week_actions.pop(0)

        assert len(self.action_plan) == 0

        cash_home = 0
        cash_work = self.u_economic_w
        virus_utility = -self._params["danger"]
        death_risk = self.death_risk
        active_infection_risk = self.work_infection_risk
        passive_infection_risk = self.home_infection_risk
        death_utility = self.u_death
        health_belief = self.p_healthy
        caution_multiplier = 100

        ################ Strategy
        ## If he hasnt gone to work 4 days in last week, must go to work
        ## If has gone to work, then compares utility of work and home
        ## Since he works minimum 4 days a week, is extra cautious about other 3 days    

        work = cash_work + virus_utility * active_infection_risk * caution_multiplier + virus_utility * (
                1 - health_belief)
        home = cash_home

        if self.last_week_actions.count("W") < 4:
            action = "W"
            logger.debug(
                    "Hasn't gone to work for 4 days in last week, so choosing {0}".format(
                            action))
            self.action_plan.append(action)
            self.last_week_actions.append(action)
            return

        elif home > work:
            action = "H"
        else:
            action = "W"

        self.action_plan.append(action)
        self.last_week_actions.append(action)

        logger.debug(
                "Perceived economic payoffs are: {0:.3f}, {1:.3f}; so choosing {2}, {3}".format(
                        work, home, self.action_plan[-1], virus_utility
                ))
