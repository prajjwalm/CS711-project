from .group import CoWorkersGame, _init as group_init
from .one_player import OnePlayerGame, _init as one_player_init


def init():
    one_player_init()
    group_init()
