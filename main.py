import logging

from enviornments import Env, init as init_environment
from games import Population, init as init_games
from players import init as init_players

if __name__ == '__main__':
    env = Env(10000, 10)


    # setup logging
    class ContextFilter(logging.Filter):
        """ Injects contextual information into the log. """

        def filter(self, record):
            record.day = env.t
            return True


    logger = logging.getLogger("Log")

    # change log level here; note: all modules use the same logger
    logger.setLevel(logging.INFO)

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

    pop = Population(env)

    # players = []
    # common = {
    #     "job_risk"      : 0.5,
    #     "job_importance": 0.1
    # }
    # for i in range(4):
    #     players.append(Coward(env, **common,
    #                           economic_status=0.4 + 0.2 * np.random.rand(),
    #                           danger=0.3 + 0.2 * np.random.rand()))
    #
    # for i in range(4):
    #     players.append(Simple(env, **common,
    #                           economic_status=0.4 + 0.2 * np.random.rand(),
    #                           danger=0.3 + 0.2 * np.random.rand()))
    #
    # for i in range(4):
    #     players.append(Planner(env, **common,
    #                            economic_status=0.4 + 0.2 * np.random.rand(),
    #                            danger=0.3 + 0.2 * np.random.rand()))
    #
    # game = CoWorkersGame(players, env)
    # game.play()
