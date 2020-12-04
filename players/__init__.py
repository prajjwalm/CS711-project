import argparse

from constants import run_once
from .base_player import BasePlayer, _init as _init_base, _add_args, _parse_args
from .coward import Coward, _init as _init_coward
from .planner import Planner, _init as _init_planner
from .simple import Simple, _init as _init_simple


def init():
    _init_base()
    _init_simple()
    _init_planner()
    _init_coward()


@run_once
def add_player_args(parser: argparse.ArgumentParser):
    player_group = parser.add_argument_group(
        title="Player Parameters",
        description="Sets the player parameters [Only relevant for single player and group games]")
    _add_args(player_group)


@run_once
def parse_player_args(args):
    _parse_args(args)
