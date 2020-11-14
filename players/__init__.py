from .base_player import _init as _init_base
from .simpleton import Simpleton, _init as _init_simpleton


def init():
    _init_base()
    _init_simpleton()
