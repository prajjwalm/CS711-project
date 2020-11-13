from typing import Dict, List, Optional
import random as rand

import numpy as np

class Environment:
    """ Implements the SIR model """

    n: int
    _s: float
    _i: float
    _r: float

    t: int
    max_t: int
    beta: float
    gamma: float

    def __init__(self, n, i, *, beta=0.05, gamma=0.01, max_t=5 * 365):
        self.n = n
        self._s = n - i
        self._i = i
        self._r = 0
        self.t = 0
        self.max_t = max_t

        self.beta = beta
        self.gamma = gamma

    def next_day(self):
        if self.t > 0:
            ds = - self.beta * self._i * self._s / self.n
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

class Person:

    # consts: in [0, 1]
    _params: Dict[str, Optional[float]] = {
        "economic_status": None,  # work desperation: (poverty)
        "danger" : None,  # danger posed by infection: (age) (health)
        "job_risk": None,  # risk of getting virus while working: (wfh etc.)
        "job_importance" : None,  # relevant to mechanism design and for groups
    }

    # TODO: CC

    # variables handled externally
    h: float  # belief over own type
    state: str
    action: str

    infected_day: int
    incubation_period: int
    recovery_period: int
    recovery_delta: int
    alpha: float = 0.07

    def __init__(self, *args, **kwargs):
        self._params.update({x:kwargs.get(x) for x in self._params})
        self.recovery_period >= 21+np.random.randint(-self.recovery_delta,self.recovery_delta)
        pass

    def work_infection_risk(self, infected_ratio):
        return infected_ratio * self._params["job_risk"] * self.alpha

    def act(self, action: str):
        # handle all consequences here
        risk = env.i / env.n + work_infection_risk(self.s / self.n) if action == "work" else 0

        if self.state == "healthy":
            # TODO: raghav
            self.h = 1 - (np.random.rand() / self.incubation_period) * (1 - env.i/env.n)
            if np.random.rand() < risk:
                self.state = "infected"
                self.infected_day = env.t

        elif self.state == "infected":
            self.h = max((1 - (env.t - self.infected_day) / self.incubation_period), 0)
            if (env.t - self.infected_day) >= self.recovery_period:
                self.state = "recovered"

        elif self.state == "recovered":
            h = 1

    def update(self, env):
        # TODO:
        if(self.work_utility > self.home_utility):
            action = "work"
        else:
            action = "home"
        self.act(action)

        # use np.random.rand()


    # TODO: revise ?
    def u_work_h(self) -> float:
        return 1 - self._params['economic_status']

    def u_home_h(self) -> float:
        return -self._params['job_importance']

    def u_loss_from_i(self) -> float:
        return self._params["danger"]

    @property
    def work_utility(self) -> float:
        return self.u_work_h() * self.h

    @property
    def home_utility(self) -> float:
        return self.u_home_h() * self.h - self.u_loss_from_i() * (1 - self.h)



if __name__ == '__main__':
    env = Environment(10000, 10)
    p = Person({
        "economic_status":
    })

    for day in env:
        print(day, env.s, env.i, env.r)
