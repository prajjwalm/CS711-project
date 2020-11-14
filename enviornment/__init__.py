from .sir import _EnvironmentSIR, _init as _init_sir


def init():
    _init_sir()


env = _EnvironmentSIR(10000, 10)
