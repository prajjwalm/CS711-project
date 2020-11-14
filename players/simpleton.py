import logging

from .base_player import _Person

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Simpleton(_Person):

    def plan(self):
        assert len(self.action_plan) == 0

        cash = self.u_economic
        self.action_plan.append(max(cash, key=cash.get))

        logger.debug("Economic payoffs are: {0}; so choosing {1}".format(
                cash, self.action_plan[-1]
        ))
