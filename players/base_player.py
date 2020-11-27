import logging
from typing import Dict, List, Optional

import numpy as np

from enviornments import BaseEnvironment

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class DeathException(Exception):
    def __init__(self, msg=None):
        self.msg = msg


class BasePlayer:
    # initialization constants
    # @formatter:off
    _params: Dict[str, Optional[float]]
    _params_list: List[str] = [
        "economic_status",  # Rich guys needn't work
        "danger",           # Danger posed by infection (age/health)
        "job_risk",         # infection risk while working (0 for wfh)
        "job_importance",   # Bonus for doctors etc. [mechanism designer]
    ]

    # reference to the global environment
    env: BaseEnvironment

    # reference to the global game

    # list of future actions [str: "W"/"H"]: All implementations fill this
    action_plan: List[str]

    # variables updated automatically
    state: str          # enum{S, I, R, X(dead)}
    p_healthy: float    # belief over own type [healthy vs infected]
    net_utility: float  # obvious
    t_i: Optional[int]  # day of infection
    t_r: Optional[int]  # day of recovery
    n_w: int            # number of days worked prior to infection

    # misc. constants
    c1: float = 1       # job risk multiplier
    c2: float = 0.03    # home risk multiplier
    c3: float = 20000   # utility loss on death
    c4: float = 0       # health inconvenience during virus
    c5: float = 1       # job importance multiplier
    c6: float = 1       # economic status multiplier

    # @formatter:on

    def __init__(self, env, *, economic_status, danger, job_risk, job_importance):
        self.env = env
        l = locals()
        self._params = {x: l.get(x) for x in self._params_list}
        self.net_utility = 0
        self.state = "S"
        self.p_healthy = 1
        self.action_plan = []
        self.t_i = None
        self.t_r = None
        self.n_w = 0

        logger.info("{0} initialized with params: {1}".format(
            self.type, self._params
        ))

    def plan(self):
        raise NotImplementedError

    # relevant only for group games
    def update(self, actions: List[str], self_idx: int):
        raise NotImplementedError

    # relevant only for group games
    def on_alert(self):
        raise NotImplementedError

    def state_change(self, risk):
        if self.state == "S" and np.random.rand() < risk:
            self.state = "I"
            self.t_i = self.env.t
        elif self.state == "I" \
                and self.env.t - self.t_i >= self.env.TIMES['removal']:
            if np.random.rand() > self.death_risk:
                self.state = "R"
                self.t_r = self.env.t
            else:
                # TODO: handle game-over here
                self.state = "X"
                self.net_utility += self.u_death
                logger.info("DEAD")
                raise DeathException

        if self.state == "R":
            self.p_healthy = 1
            return
        if self.state == "S":
            self.p_healthy = 1
        elif self.state == "I":
            self.p_healthy = 1 - (self.env.t - self.t_i) / self.env.TIMES['symptoms']
        fluctuation = 0.25
        self.p_healthy += (np.random.rand() - 0.5) * fluctuation
        self.p_healthy = min(1.0, max(0.0, self.p_healthy))

    @property
    def w_infection_risk(self):
        if self.state == "R":
            return 0
        base_infection_prob = self.env.infected_today / self.env.s
        extra_risk = base_infection_prob * self._params["job_risk"] * self.c1
        risk = self.h_infection_risk + extra_risk
        if self.state == "S":
            return risk
        else:
            assert self.state == "I"
            return risk * self.p_healthy

    @property
    def h_infection_risk(self):
        if self.state == "R":
            return 0
        base_infection_prob = self.env.infected_today / self.env.s
        risk = base_infection_prob * self.c2
        if self.state == "S":
            return risk
        else:
            assert self.state == "I"
            return risk * self.p_healthy

    @property
    def u_economic_max(self) -> float:
        return ((1 - self._params['economic_status']) * self.c5
                + self._params['job_importance'] * self.c6)

    @property
    def u_economic_w(self) -> float:
        # sick people have 0 economic utility

        # TODO: replace with mean/step (?)
        sick_reduction = self.p_healthy ** 2  # so that it falls off faster

        return self.u_economic_max * sick_reduction

    @property
    def u_virus(self) -> float:
        # inconvenience caused by virus (eg. ventilator/trauma/lung damage)
        # [unrelated to possibility of death]
        if self.state != "I":
            return 0
        if self.env.t - self.t_i > self.env.TIMES['symptoms']:
            return - self._params["danger"] * self.c4
        return 0

    @property
    def u_death(self):
        return -self.c3

    @property
    def death_risk(self) -> float:
        return 0.1 * self._params['danger'] ** 2  # TODO: may change if c4 = 0

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    def job_risk(self) -> float:
        return self._params['job_risk']

    @property
    def job_importance(self) -> float:
        return self._params['job_importance']
