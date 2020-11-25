import logging
from json import load
from typing import List

from enviornments import BaseEnvironment

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Population:
    env: BaseEnvironment

    sections: List[str]
    pops: List[float]
    params: List[List[float]]
    params_order: List[str]

    def __init__(self, env, data_file=None):
        self.env = env
        if data_file is None:
            data_file = "data.json"

        with open(data_file) as f:
            raw_data = load(f)

        self.sections = []
        self.pops = []
        self.params = []
        self.params_order = raw_data['jobs-param-order'] + ["danger"]

        for k1, v1 in raw_data["jobs"].items():
            for k2, v2 in raw_data["population"].items():
                self.sections.append(k2 + " " + k1)
                self.params.append(v1 + [v2['danger']])
                self.pops.append(v2['ratio'] * v2['job-dist'][k1])

    def coward_action(self, idx, last_w, i_status):
        """
        :param idx:      The index of the section for which this function is called {0, ..., 15}
        :type idx:       int

        :param last_w:   The ratio of people (of this type) who went to work yesterday
        :type last_w:    float

        :param i_status: Ratios for the day of infection this is (-1 -> S, t_recovery -> R)
        :type i_status:  List[float]

        :return: ratio of people who will choose to work
        :rtype:  float
        """
