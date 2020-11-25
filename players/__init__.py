from .base_player import BasePlayer, _init as _init_base, DeathException
from .coward import Coward, _init as _init_coward
from .planner import Planner, _init as _init_planner
from .simple import Simple, _init as _init_TypeR


def init():
    _init_base()
    _init_TypeR()
    _init_planner()
    _init_coward()
