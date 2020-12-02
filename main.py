import logging

from enviornments import Env, init as init_environment
from games import Population, init as init_games
from players import init as init_players


def main():
    # TODO:make this work for stage 3 too
    env = Env(10000, 10)

    # setup logging
    class ContextFilter(logging.Filter):
        """ Injects contextual information into the log. """

        def filter(self, record):
            record.day = env.t
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
    init_environment()
    init_players()
    init_games()

    pop = Population()
    pop.simulate()

if __name__ == '__main__':
    main()
