import logging

from enviornment import Env, init as init_environment
from players import Simple, init as init_players


def main():
    p = Simple(env, **{
        "economic_status": 0.4,
        "danger"         : 0.8,
        "job_risk"       : 0.5,
        "job_importance" : 0.1,
    })
    try:
        for day in env:
            p.plan()
            p.act()
            logger.info(
                    "True state: {0}, believes himself to be {1:d}% healthy, "
                    "and has a net utility of {2:.2f}, (percentage infected = "
                    "{3:.2f}%, work risk = {4:.3f}%, home risk = {5:.3f}%)"
                    "".format(
                            p.state,
                            int(p.p_healthy * 100),
                            p.net_utility,
                            env.i / env.n * 100,
                            p.w_infection_risk * 100,
                            p.h_infection_risk * 100
                    )
            )
    except NotImplementedError:
        logger.critical("Old doc dead")

    if p.t_i is not None:
        print("Went to work {0:d} days before getting infected on the "
              "{1:d}th day".format(p.n_w, p.t_i))
    else:
        print("Went to work {0:d} days, didn't get infected".format(p.n_w))


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
    logger.setLevel(logging.DEBUG)

    fh = logging.FileHandler("logs/proceeds.log", mode='w')
    fh.setFormatter(logging.Formatter(
            "%(module)s(%(lineno)d): %(funcName)s [%(levelname)s] "
            "(Day %(day)s): %(message)s"
    ))
    logger.addHandler(fh)
    logger.addFilter(ContextFilter())

    init_environment()
    init_players()

    main()
