import argparse
import logging

from constants import env_types, T
from .base_env import BaseEnvironment

logger: logging.Logger


def _add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--n-start", type=int, metavar="n", default=100000, help="#Initial total population")
    parser.add_argument("--i-start", type=int, metavar="i", default=10, help="#Initial infected population")


def _parse_args(args: argparse.Namespace):
    # t_max added by base env
    return _EnvironmentSIR(args.n_start, args.i_start, max_t=args.t_max)


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class _EnvironmentSIR(BaseEnvironment):
    """ Implements the SIR model """

    _n: int
    _s: float
    _i: float
    _r: float

    t: int
    max_t: int
    beta: float
    gamma: float

    def __init__(self, n, i, max_t, beta=env_types['sir']['beta'], gamma=env_types['sir']['gamma']):
        super().__init__(n, max_t=max_t)
        self._s = n - i
        self._i = i
        self._r = 0

        self.beta = self.R0 / self.TIMES["removal"] if beta is None else beta
        self.gamma = 1 / self.TIMES["removal"] if gamma is None else gamma

    def next_day(self):
        if T[0] > 0:
            ds = - self.infected_today
            dr = self.gamma * self._i
            di = - ds - dr
            self._s += ds
            self._i += di
            self._r += dr
        T[0] += 1

    @property
    def s(self) -> int:
        return self._n - int(self._i) - int(self._r)

    @property
    def e(self) -> int:
        return 0

    @property
    def i(self) -> int:
        return int(self._i)

    @property
    def r(self) -> int:
        return int(self._r)

    @property
    def infected_today(self):
        return self.beta * self._i * self._s / self._n
