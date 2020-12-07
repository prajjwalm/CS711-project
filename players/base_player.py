import argparse
import logging
from typing import List, Optional

import numpy as np

from constants import sections, max_utility, job_risk, survival, player_data
from enviornments import BaseEnvironment

logger: logging.Logger
section_idx: int = 0
ages = ["y", "m", "o"]
professions = ["p", "s", "t", "e", "r"]


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


# noinspection SpellCheckingInspection
def _add_args(parser):
    parser.add_argument("--age-group", choices=ages, default="y",
                        help="Age bracket of the person/group members ([y]oung: < 45 yrs old, "
                             "[m]iddle: 45 to 60 yrs old, [o]ld: > 60 yrs old)")
    parser.add_argument("--job-type", choices=professions, default="p",
                        help="The nature of the profession of the individual/group members. "
                             "[p]rimary/[s]econdary/[t]ertiary are the three sectors; [e]ssential "
                             "denotes jobs like doctors/police that are important in emergencies "
                             "like pandemics and [r]etired denotes people that no longer need work")


def _parse_args(args: argparse.Namespace):
    global section_idx
    try:
        section_idx = (1 + ages.index(args.age_group)) * (1 + professions.index(args.job_type)) - 1
    except ValueError:
        logger.warning("Invalid age/profession provided, going with default (young/primary)")


class BasePlayer:
    # initialization constants
    # @formatter:off

    class DeathException(Exception):
        def __init__(self, msg=None):
            self.msg = msg

    # key to all parametric information
    section_idx: int

    # reference to the global environment
    env: BaseEnvironment

    # list of future actions [str: "W"/"H"]: All implementations fill this
    action_plan: List[str]

    # variables updated automatically
    state: str          # enum{S, I, R, X(dead)}
    p_healthy: float    # belief over own type [healthy vs infected]
    net_utility: float  # obvious
    t_i: Optional[int]  # day of infection
    t_r: Optional[int]  # day of recovery
    n_w: int            # number of days worked prior to infection

    # @formatter:on

    def __init__(self, env):
        self.env = env

        loc = locals()
        self.section_idx = section_idx
        assert 0 <= self.section_idx < len(sections)

        self.net_utility = 0
        self.state = "S"
        self.p_healthy = 1
        self.action_plan = []
        self.t_i = None
        self.t_r = None
        self.n_w = 0

        logger.info("{0} {1} initialized with params".format(sections[self.section_idx], self.type))

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
            dice = np.random.rand()
            logger.debug("P[death]: {0:f}, dice: {1:f}".format(self.death_risk, dice))
            if dice > self.death_risk:
                self.state = "R"
                self.t_r = self.env.t
            else:
                self.state = "X"
                self.net_utility += self.u_death
                logger.info("DEAD")
                raise self.DeathException

        if self.state == "R":
            self.p_healthy = 1
            return
        if self.state == "S":
            self.p_healthy = 1
        elif self.state == "I":
            self.p_healthy = 1 - (self.env.t - self.t_i) / self.env.TIMES['symptoms']
        fluctuation = player_data['p-healthy-fluctuation']
        self.p_healthy += (np.random.rand() - 0.5) * fluctuation
        self.p_healthy = min(1.0, max(0.0, self.p_healthy))

    @property
    def w_infection_risk(self):
        if self.state == "R":
            return 0
        base_infection_prob = self.env.infected_today / self.env.s
        extra_risk = base_infection_prob * job_risk[self.section_idx] * player_data['x-work-risk']
        # TODO p healthy is multiplied twice in w infection risk.
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
        risk = base_infection_prob * player_data['x-home-risk']
        if self.state == "S":
            return risk
        else:
            assert self.state == "I"
            return risk * self.p_healthy

    @property
    def u_economic_max(self) -> float:
        return max_utility[self.section_idx]

    @property
    def u_economic_w(self) -> float:
        return self.u_economic_max * self.p_healthy ** 2

    @property
    def u_death(self):
        return -player_data['u-death']

    @property
    def death_risk(self) -> float:
        return 1 - survival[self.section_idx]

    @property
    def type(self) -> str:
        return self.__class__.__name__

    @property
    def job_risk(self) -> float:
        return job_risk[self.section_idx]

    @property
    def pW(self) -> float:
        raise NotImplementedError
