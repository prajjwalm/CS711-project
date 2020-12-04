from constants import run_once
from enviornments import add_env_args, parse_env_args
from players import add_player_args, parse_player_args
from .group_game import _init as _init_2, _add_args as _add_2, _parse_args as _parse_2, GroupGame
from .one_player import _init as _init_1, _add_args as _add_1, _parse_args as _parse_1, OnePlayer
from .population import _init as _init_3, _add_args as _add_3, _parse_args as _parse_3, Population

_parse = {
    "person"    : _parse_1,
    "group"     : _parse_2,
    "population": _parse_3,
}


def init():
    _init_1()
    _init_2()
    _init_3()


@run_once
def add_game_args(parser):
    add_player_args(parser)
    add_env_args(parser)
    games = parser.add_argument_group(
        title="Game Participants",
        description="Set the information of how many of each player type [where each type denotes "
                    "a certain play-style] to set in the game")

    _add_1(games)
    _add_2(games)
    _add_3(games)


@run_once
def parse_game_args(args):
    if args.simulate != "population":
        parse_player_args(args)
        env = parse_env_args(args)
        return _parse[args.simulate](args, env)
    return _parse[args.simulate](args)
