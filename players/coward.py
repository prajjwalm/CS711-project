import logging
from typing import List

from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Coward(BasePlayer):
    last_action: str  # Logs last action

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_action = "W"

    def plan(self):
        """
        Strategy:
        Doesn't care about utilities. Doesn't plan into the future. Works if
        feeling healthy today, else stays at home.
        [TODO: affect threshold with current number of cases?]
        """

        assert len(self.action_plan) == 0

        threshold = 0.9 if self.last_action == "H" else 0.96

        if self.p_healthy > threshold:
            self.action_plan.append("W")
        else:
            self.action_plan.append("H")

        self.last_action = self.action_plan[-1]

        logger.debug("Health belief is {0:.2f}, so choosing {1}".format(
                self.p_healthy, self.action_plan[-1]
        ))

    def update(self, actions: List[str], self_idx: int):
        pass
