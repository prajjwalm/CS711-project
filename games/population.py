import logging
from json import load
from typing import List, Tuple, Dict

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
    fluctuation: float

    coward_data: Dict

    def __init__(self, env, data_file=None):
        self.env = env
        if data_file is None:
            data_file = "data.json"

        with open(data_file) as f:
            raw_data = load(f)

        self.fluctuation = raw_data['p-healthy-fluctuation']
        self.sections = []
        self.pops = []
        self.params = []
        self.params_order = raw_data['jobs-param-order'] + ["danger"]

        self.coward_data = raw_data["player-types"]["coward"]

        for k1, v1 in raw_data["jobs"].items():
            for k2, v2 in raw_data["population"].items():
                self.sections.append(k2 + " " + k1)
                self.params.append(v1 + [v2['danger']])
                self.pops.append(v2['ratio'] * v2['job-dist'][k1])

    def coward_action(self, idx, last_w, i_status):
        """
        :param idx:      The index of the section for which this function is called {0, ..., 15}[UNUSED]
        :type idx:       int

        :param last_w:   The ratio of people (of this type) who went to work yesterday for each day
                         of infection
                         e.g. idx = 0, is the ratio (young primary coward sus W(t-1) / young primary cowards sus)
        :type last_w:    List[float]

        :param i_status: Ratios for the day of infection this is (0 -> S, t_recovery -> R)
                         e.g. idx = 0, is the ratio (young primary coward sus / young primary cowards)
        :type i_status:  List[float]

        :return: ratio of people who will choose to work
        :rtype:  float
        """
        w = []
        threshold_sw = self.coward_data['w-threshold']
        threshold_sh = self.coward_data['h-threshold']

        total_days = self.env.TIMES["removal"]
        unaware_days = self.env.TIMES["infectious"]

        ratio_over_threshold_w = 1 - ((threshold_sw - (1 - self.fluctuation / 2)) / self.fluctuation)
        ratio_over_threshold_h = 1 - ((threshold_sh - (1 - self.fluctuation / 2)) / self.fluctuation)

        w.append(last_w[0] * ratio_over_threshold_w + (1 - last_w[0]) * ratio_over_threshold_h)

        for i in range(total_days):
            if i < unaware_days:
                ratio_over_threshold_w = 1 - (
                        (threshold_sw - (1 - (i + 1) / unaware_days - self.fluctuation / 2)) / self.fluctuation)
                ratio_over_threshold_w = min(1, max(ratio_over_threshold_w, 0))
                ratio_over_threshold_h = 1 - (
                        (threshold_sh - (1 - (i + 1) / unaware_days - self.fluctuation / 2)) / self.fluctuation)
                ratio_over_threshold_h = min(1, max(ratio_over_threshold_h, 0))
            else:
                ratio_over_threshold_w = max((self.fluctuation / 2 - threshold_sw) / self.fluctuation, 0)
                ratio_over_threshold_h = max((self.fluctuation / 2 - threshold_sh) / self.fluctuation, 0)
            w.append(last_w[i] * ratio_over_threshold_w + (1 - last_w[i]) * ratio_over_threshold_h)

        ratio_over_threshold_w = 1 - ((threshold_sw - (1 - self.fluctuation / 2)) / self.fluctuation)
        ratio_over_threshold_h = 1 - ((threshold_sh - (1 - self.fluctuation / 2)) / self.fluctuation)

        w.append(last_w[total_days] * ratio_over_threshold_w + (1 - last_w[total_days]) * ratio_over_threshold_h)

        return w
