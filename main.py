import argparse
import logging

import numpy as np

from constants import T
from enviornments import init as init_env
from games import init as init_game, add_game_args, parse_game_args
from players import init as init_players


def add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--simulate", choices=["person", "group", "population"], default="population",
                        help="Selecting person/group runs a simulation on one person/a group of co-"
                             "workers in an environment governed by the SIR model, whereas choosing "
                             "population attempts to simulate the virus (again using the SIR model) "
                             "and utilities on the entire population using densities")

    parser.add_argument("--plot", help="Plot graphs", action="store_true")
    parser.add_argument("--debug", help="proceeds.log runs with DEBUG mode", action="store_true")
    add_game_args(parser)


def parse_args(args: argparse.Namespace):
    plot = args.plot
    debug = args.debug
    return plot, debug


def main():
    # setup logging
    class ContextFilter(logging.Filter):
        """ Injects contextual information into the log. """

        def filter(self, record):
            record.day = T[0]
            return True

    logger = logging.getLogger("Log")

    # setup logging for imported modules
    init_env()
    init_players()
    init_game()

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    args = parser.parse_args()
    plot, debug = parse_args(args)

    # change log level here; note: all modules use the same logger
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    # logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/proceeds.log", mode='w')
    fh.setFormatter(logging.Formatter(
        "%(module)s(%(lineno)d): %(funcName)s [%(levelname)s] "
        "(Day %(day)s): %(message)s"
    ))
    logger.addHandler(fh)
    logger.addFilter(ContextFilter())

    np.set_printoptions(precision=3, linewidth=300)

    game = parse_game_args(args)
    game.simulate()
    if plot:
        game.plot_graphs()


if __name__ == '__main__':
    main()
    # one_player_main()
