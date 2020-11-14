import logging
from typing import Dict, List, Optional

import numpy as np

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class _Person:
    # initialization constants
    # @formatter:off
    _params: Dict[str, Optional[float]]
    _params_list: List[str] = [
        "economic_status",  # Rich guys needn't work
        "danger",           # Danger posed by infection (age/health)
        "job_risk",         # infection risk while working (0 for wfh)
        "job_importance",   # Penalty for not working (so doctors work)
    ]

    # reference to the global environment

    # list of future actions [str: "W"/"H"]: All implementations fill this
    action_plan: List[str]

    # variables updated automatically
    state: str          # enum{S, I, R, X(dead)}
    p_healthy: float    # belief over own type [healthy vs infected]
    net_utility: float  # obvious
    t_i: int            # day of infection
    t_r: int            # day of recovery

    # misc. constants
    c1: float = 0.07  # job risk multiplier
    c2: float = 1000  # utility loss on death

    # @formatter:on

    def __init__(self, env, *args, **kwargs):
        self.env = env
        self._params = {x: kwargs.get(x) for x in self._params_list}
        self.net_utility = 0
        self.state = "S"
        self.belief_update()
        self.action_plan = []

        logger.debug("{0} initialized with params: {1}".format(
                self.type, self._params
        ))

    def plan(self):
        raise NotImplementedError

    def act(self):
        action = self.action_plan.pop()
        self.net_utility += self.u_economic[action] + self.u_virus

        risk = self.env.i / self.env.n
        if action == "W":
            risk += self.work_infection_risk

        self.state_change(risk)
        self.belief_update()

    def state_change(self, risk):
        if self.state == "S" and np.random.rand() < risk:
            self.state = "I"
            self.t_i = self.env.t
        elif self.state == "I" and self.env.t - self.t_i >= self.env.t_recovery:
            if np.random.rand() > self.death_risk:
                self.state = "R"
                self.t_r = self.env.t
            else:
                # TODO: handle game-over here
                self.state = "X"
                self.net_utility += self.u_death
                raise NotImplementedError

    def belief_update(self):
        """ results from a person's observation of their own symptoms """
        if self.state == "R":
            self.p_healthy = 1
            return

        # TODO
        if self.state == "S":
            self.p_healthy = 1
        elif self.state == "I":
            self.p_healthy = 1 - (self.env.t - self.t_i) / self.env.t_incubation
        self.p_healthy += np.random.normal() / 10
        self.p_healthy = min(1.0, max(0.0, self.p_healthy))

    @property
    def work_infection_risk(self):
        if self.state != "S":
            return 0
        return self.env.s / self.env.n * self._params["job_risk"] * self.c1

    @property
    def u_economic(self) -> Dict[str, float]:
        # sick people have 0 economic utility
        return {
            "W": (1 - self._params['economic_status']) * self.p_healthy,
            "H": (-self._params['job_importance']) * self.p_healthy
        }

    @property
    def u_virus(self) -> float:
        # inconvenience caused by virus (eg. ventilator/trauma/lung damage)
        # [unrelated to possibility of death]
        if self.state != "I":
            return 0
        if self.env.t - self.t_i > self.env.t_incubation:
            return - self._params["danger"]
        return 0

    @property
    def u_death(self):
        return -self.c2

    @property
    def death_risk(self) -> float:
        return self._params['danger'] ** 2  # TODO: may change

    @property
    def type(self) -> str:
        return self.__class__.__name__