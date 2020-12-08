import logging
from typing import List

import numpy as np

from constants import player_types, survival, player_data
from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Planner(BasePlayer):
    last_week_actions: List[str]
    target_days: int
    target_ratio: float
    h: float = player_types['planner']['h']
    X: float

    work_weightage: np.ndarray

    def __init__(self, env):
        super().__init__(env)
        self.X = 1 / (1 - self.h)
        self.alert = False
        self._pW = 1

    def plan(self):
        assert len(self.action_plan) == 0
        uv = (self.w_infection_risk - self.h_infection_risk) * (1 - survival[self.section_idx]) * player_data['u-death']
        ue = self.u_economic_max
        cutoff_ss = np.sqrt(uv / ue)
        p_h_mean = 1 if self.t_i is None else 1 - (self.env.t - self.t_i) / self.env.TIMES['symptoms']
        p_h_delta = player_data['p-healthy-fluctuation']

        if self.state == "R":
            logger.debug("Working as recovered")
            self.action_plan.append("W")
            self._pW = 1
            return

        logger.debug("Simple Sus Cutoff: {0:.2f}".format(cutoff_ss))
        if cutoff_ss > 1:
            action = "H"
            self.X = 0 + self.X * self.h
            self._pW = 0
        else:
            p_w_mean = (p_h_mean + p_h_delta / 2 - cutoff_ss) / p_h_delta
            p_w_mean = min(1, max(0, p_w_mean))
            p_w_del = min(1 - p_w_mean, p_w_mean) * player_types['planner']['cap']
            p_w_min = p_w_mean - p_w_del
            p_w_max = p_w_mean + p_w_del
            assert 0 <= p_w_min <= p_w_max <= 1
            p_w = (p_w_min - p_w_max) * (1 - self.h) * self.X + p_w_max
            logger.debug("P[H] in [{0:.2f}, {1:.2f}]".format(p_h_mean - p_h_delta / 2, p_h_mean + p_h_delta / 2))
            logger.debug("SS p[w] = {0:.2f}, p_max[w] = {1:.2f}, p_min[w] = {2:.2f}, X = {3:.2f}, p[w] = {4:.2f}"
                         "".format(p_w_mean, p_w_max, p_w_min, self.X, p_w))
            self._pW = p_w
            if np.random.rand() < p_w:
                action = 'W'
                self.X = 1 + self.X * self.h
            else:
                action = "H"
                self.X = 0 + self.X * self.h
        logger.debug("Action: {0}, X: {1:.2f}".format(action, self.X))

        self.action_plan.append(action)

    def update(self, actions: List[str], self_idx: int):
        self.alert = False

    def on_alert(self):
        self.alert = True

    @property
    def pW(self) -> float:
        return self._pW
