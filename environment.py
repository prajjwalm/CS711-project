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
    healthy: float
    _params: Dict[str, float] = {
        "economic_status": 0.0,  # if can survive without working
        "job_importance" : 0.0,  # like doctor
        "danger" : 0.0,  # possible consequences upon getting infected
        "job_risk": 0.0,  # risk of getting virus while working
    }

    def __init__(self, *args, **kwargs: Dict[str, float]):
        self._params.update({x:kwargs.get(x) for x in self._params})
        pass


    @property
    def work_utility(self) -> float:
        pass

    @property
    def home_utility(self) -> float:
        pass



if __name__ == '__main__':
    env = Environment(10000, 10)

    for day in env:
        print(day, env.s, env.i, env.r)
