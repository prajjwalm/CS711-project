import logging
from typing import List

from enviornments import BaseEnvironment
from players import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class GroupGame:
    players: List[BasePlayer]
    utilities: List[float]
    risks: List[float]
    actions: List[str]

    env: BaseEnvironment

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        self.env = env
        self.players = players
        self.utilities = [0] * len(players)
        self.risks = [0] * len(players)
        self.actions = ["W"] * len(players)

    def handle_utilities(self):
        raise NotImplementedError

    def handle_risk(self):
        raise NotImplementedError

    def alert(self):
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

                # if someone is symptomatic
                self.alert()

                # inform players of other's actions
                for i in range(len(self.players)):
                    self.players[i].update(self.actions, i)

                # log
                logger.info(
                    "\n States:        {3}\n Actions taken: {0}\n Utilities:     {1}\n Risks (%):     {2}".format(
                        " ".join(["   {:4}".format(x) for x in self.actions]),
                        " ".join(["{:7.2f}".format(x) for x in self.utilities]),
                        " ".join(["{:7.3f}".format(x * 100) for x in self.risks]),
                        " ".join(["   {:4}".format(x.state) for x in self.players])
                    ))

                if n_days is not None and day == n_days:
                    break
        except NotImplementedError:
            logger.critical("Group member dead")


class CoWorkersGame(GroupGame):

    job_risk: float
    job_importance: float

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        super().__init__(players, env)

        # all players share the same job
        self.job_risk = players[0].job_risk
        for player in players[1:]:
            assert player.job_risk == self.job_risk

        self.job_importance = players[0].job_importance
        for player in players[1:]:
            assert player.job_importance == self.job_importance

    def alert(self):
        symptoms_cutoff = self.env.t - self.env.TIMES['symptoms']
        alert = False
        for p in self.players:
            if p.state == "I" and p.t_i < symptoms_cutoff:
                alert = True
                break
        if alert:
            for p in self.players:
                p.on_alert()

    def handle_utilities(self):
        n_work = self.actions.count("W")
        if n_work == 0:
            return
        max_utility = self.players[0].u_economic_max * len(self.players)
        net_utility = 0
        for i in range(len(self.players)):
            if self.actions[i] == 'W':
                net_utility += self.players[i].u_economic_w
        u_per_person = net_utility * net_utility / (n_work * max_utility)
        for i in range(len(self.players)):
            if self.actions[i] == "W":
                self.utilities[i] += u_per_person

    def handle_risk(self):
        n_infectious = 0
        infectious_cutoff = self.env.t - self.env.TIMES["infectious"]
        for player in self.players:
            if player.state == "I" and player.t_i < infectious_cutoff:
                n_infectious += 1

        for i in range(len(self.players)):
            if self.actions[i] == "H":
                risk = self.players[i].h_infection_risk
            elif n_infectious == 0:
                risk = self.players[i].w_infection_risk
            else:
                # probability of infection when one is sick
                pr = self.env.R0 / (self.env.TIMES["removal"] - self.env.TIMES['infectious'])

                # total probability of infection
                risk = (1 - (1 - pr) ** n_infectious) * self.job_risk + self.players[i].w_infection_risk
            self.players[i].state_change(risk)
            self.risks[i] = risk


class NeighboursGame(GroupGame):

    def __init__(self, players: List[BasePlayer], env: BaseEnvironment):
        super().__init__(players, env)

    def handle_utilities(self):
        pass

    def handle_risk(self):
        pass
