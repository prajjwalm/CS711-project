import argparse
import logging
from typing import Dict

from constants import env_params, T

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


def _add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--t-max", type=int, metavar="T", default=365, help="Simulation Time Period")


class BaseEnvironment:
    """ All models inherit from this """

    # consts
    TIMES: Dict[str, int] = {
        "infectious": env_params['t-infectious'],
        "symptoms"  : env_params['t-symptoms'],
        "removal"   : env_params['t-removal']
    }
    R0: float = env_params['R0']

    def __init__(self, n, max_t):
        self._n = n
        self.max_t = max_t
        assert T[0] == 0

    def next_day(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if T[0] == self.max_t:
            raise StopIteration
        self.next_day()
        return T[0]

    @property
    def n(self) -> int:
        return self._n

    @property
    def s(self) -> int:
        raise NotImplementedError

    @property
    def e(self) -> int:
        raise NotImplementedError

    @property
    def i(self) -> int:
        raise NotImplementedError

    @property
    def r(self) -> int:
        raise NotImplementedError

    @property
    def t(self) -> int:
        return T[0]

    @property
    def infected_today(self):
        raise NotImplementedError
