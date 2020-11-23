import logging
from typing import List

from enviornment import BaseEnvironment
from players import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class GroupGame:
    players: List[BasePlayer]
    utilities: List[float]
    actions: List[str]

    env: BaseEnvironment

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        self.env = env
        self.players = players
        self.utilities = [0] * len(players)
        self.actions = ["W"] * len(players)

    def handle_utilities(self):
        raise NotImplementedError

    def handle_risk(self):
        raise NotImplementedError

    def play(self, n_days=None):
        try:
            for day in self.env:
                # allow players to plan
                for player in self.players:
                    player.plan()

                # get player actions
                for i in range(len(self.players)):
                    self.actions[i] = self.players[i].action_plan.pop()

                # handle utilities
                self.handle_utilities()

                # handle risk
                self.handle_risk()

                # inform players of other's actions
                for i in range(len(self.players)):
                    self.players[i].update(self.actions, i)

                # log
                # nothing

                if n_days is not None and day == n_days:
                    break
        except NotImplementedError:
            logger.critical("Group member dead")


class CoWorkersGame(GroupGame):

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        super().__init__(players, env)

        # all players share the same job
        group_job_risk = players[0].job_risk
        for player in players[1:]:
            assert player.job_risk == group_job_risk

        group_job_importance = players[0].job_importance
        for player in players[1:]:
            assert player.job_importance == group_job_importance

    def handle_utilities(self):
        pass

    def handle_risk(self):
        pass


class NeighboursGame(GroupGame):

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        super().__init__(players, env)

    def handle_utilities(self):
        pass

    def handle_risk(self):
        pass
