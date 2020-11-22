from .base_player import _init as _init_base
from .type_papa import TypePapa, _init as _init_TypePapa
from .type_r import TypeR, _init as _init_TypeR
from .type_ri import TypeRi, _init as _init_TypeRi
from .type_shr import Planner, _init as _init_planner


def init():
    _init_base()
    _init_TypeR()
    _init_TypeRi()
    _init_planner()
    _init_TypePapa()
