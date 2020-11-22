from .base_player import _init as _init_base
from .coward import Coward, _init as _init_coward
# from .type_ri import TypeRi, _init as _init_TypeRi
from .planner import Planner, _init as _init_planner
from .simple import Simple, _init as _init_TypeR


def init():
    _init_base()
    _init_TypeR()
    # _init_TypeRi()
    _init_planner()
    _init_coward()
