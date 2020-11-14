from .base_player import _init as _init_base
from .simpleton import Simpleton, _init as _init_simpleton
from .TypeR import TypeR, _init as _init_TypeR

def init():
    _init_base()
    _init_simpleton()
    _init_TypeR()