import logging
from typing import Dict

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class BaseEnvironment:
    """ All models inherit from this """

    # consts
    TIMES: Dict[str, int] = {
        "infectious": 3,  # a person becomes capable of infecting others here
        "symptoms"  : 5,  # from this time a person starts showing symptoms
        "removal"   : 21  # expected recovery time (for normal cases)
    }
    R0: float = 2.4

    def __init__(self, n, *, max_t=5 * 365):
        # THIS CONSTRUCTOR CANNOT HAVE LOGGING
        self._n = n
        self.t = 0
        self.max_t = max_t

    def next_day(self):
        raise NotImplementedError

    def __iter__(self):
        return self

    def __next__(self):
        if self.t == self.max_t:
            raise StopIteration
        self.next_day()
        return self.t

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
    def infected_today(self):
        raise NotImplementedError
