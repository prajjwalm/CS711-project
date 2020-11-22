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
    t_w: int            # first day of work
    n_w: int            # number of days worked

    # misc. constants
    c1: float = 1       # job risk multiplier
    c2: float = 0.1     # home risk multiplier
    c3: float = 500     # utility loss on death
    c4: float = 1       # health inconvenience during virus
    c5: float = 1       # job importance multiplier
    c6: float = 1       # economic status multiplier


    # @formatter:on

    def __init__(self, env, *args, **kwargs):
        self.env = env
        self._params = {x: kwargs.get(x) for x in self._params_list}
        self.net_utility = 0
        self.state = "S"
        self.belief_update()
        self.action_plan = []
        self.t_w = None
        self.t_i = None
        self.n_w = 0

        logger.debug("{0} initialized with params: {1}".format(
                self.type, self._params
        ))

    def plan(self):
        raise NotImplementedError

    def act(self):
        action = self.action_plan.pop()
        self.net_utility += \
            self.u_economic_w if action == 'W' else 0 + self.u_virus

        risk = self.home_infection_risk
        if action == "W":
            risk += self.work_infection_risk

            if self.t_w is None:
                self.t_w = self.env.t
            elif self.t_i is None:
                self.n_w += 1

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
        return self.env.infected_today / self.env.n * self._params[
            "job_risk"] * self.c1

    @property
    def home_infection_risk(self):
        if self.state != "S":
            return 0
        return self.env.infected_today / self.env.n * self.c2

    @property
    def u_economic_w(self) -> float:
        # sick people have 0 economic utility
        return ((1 - self._params['economic_status']) * self.c5
                + self._params['job_importance'] * self.c6) * self.p_healthy

    @property
    def u_virus(self) -> float:
        # inconvenience caused by virus (eg. ventilator/trauma/lung damage)
        # [unrelated to possibility of death]
        if self.state != "I":
            return 0
        if self.env.t - self.t_i > self.env.t_incubation:
            return - self._params["danger"] * self.c4
        return 0

    @property
    def u_death(self):
        return -self.c3

    @property
    def death_risk(self) -> float:
        return 0.1 * self._params['danger'] ** 2  # TODO: may change

    @property
    def type(self) -> str:
        return self.__class__.__name__
