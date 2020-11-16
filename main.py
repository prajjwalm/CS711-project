import logging

from enviornment import env, init as init_environment
from players import TypeR, typeShr, Simpleton, init as init_players


def main():
    old_doc = typeShr(env, **{
        "economic_status": 0.42,
        "danger"         : 0.7,
        "job_risk"       : 0.5,
        "job_importance" : 0.1,
    })

    try:
        for day in env:
            old_doc.plan()
            old_doc.act()
            logger.info(
                    "True state: {0}, believes himself to be {1:d}% healthy, "
                    "and has a net utility of {2:.2f}".format(
                            old_doc.state,
                            int(old_doc.p_healthy * 100),
                            old_doc.net_utility
                    )
            )
    except NotImplementedError:
        logger.critical("Old doc dead")


if __name__ == '__main__':
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
