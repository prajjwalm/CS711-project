from constants import run_once
from .base_env import BaseEnvironment, _add_args as _add_base
from .sir import _EnvironmentSIR as EnvSir, _init as _init_sir, _add_args as _add_sir, _parse_args as _parse_sir


def init():
    _init_sir()


@run_once
def add_env_args(parser):
    env = parser.add_argument_group(
        title="Environment Parameters",
        description="Initialize the environment [the population based game will only consider ratios"
                    "of populations]")

    _add_base(env)
    _add_sir(env)


@run_once
def parse_env_args(args):
    return _parse_sir(args)
