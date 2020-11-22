from .sir import _EnvironmentSIR as EnvSIR, _init as _init_sir


def init():
    _init_sir()


env = EnvSIR(10000, 10)
