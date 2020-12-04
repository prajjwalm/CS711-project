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
    add_game_args(parser)


def parse_args(args: argparse.Namespace):
    game = parse_game_args(args)
    return game


def main():
    # setup logging
    class ContextFilter(logging.Filter):
        """ Injects contextual information into the log. """

        def filter(self, record):
            record.day = T[0]
            return True

    logger = logging.getLogger("Log")

    # change log level here; note: all modules use the same logger
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/proceeds.log", mode='w')
    fh.setFormatter(logging.Formatter(
        "%(module)s(%(lineno)d): %(funcName)s [%(levelname)s] "
        "(Day %(day)s): %(message)s"
    ))
    logger.addHandler(fh)
    logger.addFilter(ContextFilter())

    # setup logging for imported modules
    init_env()
    init_players()
    init_game()

    # noinspection PyTypeChecker
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_args(parser)
    game = parse_args(parser.parse_args())

    np.set_printoptions(precision=2, linewidth=240)

    game.simulate()
    game.daily_utility_plot()

if __name__ == '__main__':
    main()
