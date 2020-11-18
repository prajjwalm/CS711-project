import logging
from typing import Dict

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class _EnvironmentSIR:
    """ Implements the SIR model """

    n: int
    _s: float
    _i: float
    _r: float

    t: int
    max_t: int
    beta: float
    gamma: float

    # consts
    # TODO: someone get reliable data
    TIMES: Dict[str, float] = {
        "infectious": 3,  # a person becomes capable of infecting others here
        "symptoms"  : 5,  # from this time a person starts showing symptoms
        "removal"   : 21  # expected recovery time (for normal cases)
    }
    R0: float = 1.5

    def __init__(self, n, i, *, beta=None, gamma=None, max_t=5 * 365):
        # THIS CONSTRUCTOR (AND ONLY THIS CONSTRUCTOR) CANNOT HAVE LOGGING
        self.n = n
        self._s = n - i
        self._i = i
        self._r = 0
        self.t = 0
        self.max_t = max_t

        self.beta = self.R0 / self.TIMES["removal"] if beta is None else beta
        self.gamma = 1 / self.TIMES["removal"] if gamma is None else gamma

    def next_day(self):
        if self.t > 0:
            ds = - self.infected_today
            dr = self.gamma * self._i
            di = - ds - dr
            self._s += ds
            self._i += di
            self._r += dr
        self.t += 1

    def __iter__(self):
        return self

    def __next__(self):
        if self.t == self.max_t:
            raise StopIteration
        self.next_day()
        return self.t

    @property
    def s(self):
        return self.n - int(self._i) - int(self._r)

    @property
    def i(self):
        return int(self._i)

    @property
    def r(self):
        return int(self._r)

    @property
    def t_incubation(self):
        return self.TIMES['symptoms']

    @property
    def t_recovery(self):
        return self.TIMES['removal']

    @property
    def infected_today(self):
        return self.beta * self._i * self._s / self.n
