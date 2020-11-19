from .base_player import _init as _init_base
from .simpleton import Simpleton, _init as _init_simpleton
from .TypeR import TypeR, _init as _init_TypeR
from .TypeRi import TypeRi, _init as _init_TypeRi
from .typeShr import TypeShr, _init as _init_TypeShr
from .TypePapa import TypePapa, _init as _init_TypePapa


def init():
    _init_base()
    _init_simpleton()
    _init_TypeR()
    _init_TypeRi()
    _init_TypeShr()
    _init_TypePapa()
