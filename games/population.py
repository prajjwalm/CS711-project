import logging
from json import load
from typing import List, Union, Dict

import numpy as np

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Population:
    # @formatter:off

    # the sequence in which player parameters are stored
    params_order:   List[str]

    # each section denotes a subset of the population approximated with the same parameters
    sections:       List[str]           # the names describing each of the sections
    pops:           List[float]         # population ratios of each of the sections (initial)
    params:         List[List[float]]   # the parameters for each section

    # size range over which a persons belief of his own health (type) varies
    fluctuation:    float

    # constants used by the environments
    env_params:     Dict[str, Union[int, float]]

    # constants used in various archetypes
    coward_data:    Dict[str, float]

    # player division across ( section x archetype x i_stage )
    n_stages:       int         # number of stages
    people:         np.ndarray  # ratio of population for each cell
    eta_w:          np.ndarray  # ratio of people who worked / will work in that cell

    deaths:         np.ndarray  # ( section x archetype )

    # prototype enum
    C:              int = 0     # archetype index for type coward
    P:              int = 1     # archetype index for type planner
    S:              int = 2     # archetype index for type simple
    N:              int = 3     # number of archetypes

    # @formatter:on

    def __init__(self, data_file=None):
        if data_file is None:
            data_file = "data.json"

        with open(data_file) as f:
            raw_data = load(f)

        self.sections = []
        self.pops = []
        self.params = []
        self.params_order = raw_data['jobs-param-order'] + ["danger"]
        self.env_params = raw_data['game-params']
        self.fluctuation = self.env_params['p-healthy-fluctuation']
        self.coward_data = raw_data["player-types"]["coward"]

        for k1, v1 in raw_data["jobs"].items():
            for k2, v2 in raw_data["population"].items():
                self.sections.append(k2 + " " + k1)
                self.params.append(v1 + [v2['danger']])
                self.pops.append(v2['ratio'] * v2['job-dist'][k1])

        self.n_stages = self.env_params['t-removal'] + 1
        self.people = np.zeros((len(self.sections), 3, self.n_stages))
        self.people[:, self.C, 0] = self.pops
        print(self.people)

    def execute_infection(self, eta_iw, eta_ih):
        """
        :param eta_iw: ratio of people who got infected among those who worked yesterday
        :type eta_iw:  float

        :param eta_ih: ratio of people who got infected among those who were at home yesterday
        :type eta_ih:  float
        """

        # 1. find out the ratio of susceptible that are getting infected
        eta_ws = self.eta_w[:, :, 0]  # ratio of people who worked (section x archetype)
        eta_i = eta_ws * eta_iw + (1 - eta_ws) * eta_ih  # ratio of people who got infected (section x archetype)
        assert eta_i.shape == (len(self.sections), self.N)

        # 2. find out the ratio of last day infected recovering
        d_idx = 3
        assert self.params_order[d_idx] == "danger"
        survival = 1 - np.asarray([0.1 * x[d_idx] ** 2 for x in self.params])
        assert survival.shape == (len(self.sections), self.N)

        # 3. execute the transitions
        fresh_infected = eta_i * self.people[:, :, 0]
        fresh_recovered = survival * self.people[:, :, -2]
        fresh_deaths = self.people[:, :, -2] - fresh_recovered

        self.people[:, :, 2:-1] = self.people[:, :, 1:-2]
        self.people[:, :, -1] += fresh_recovered
        self.people[:, :, 0] -= fresh_infected
        self.people[:, :, 1] = fresh_infected
        self.deaths += fresh_deaths

    def execute_action_cowards(self):
        pass

    # TODO(@CC): do this directly using numpy (i.e. remove loops), handle all idx together
    # should also be able to handle all i_stages together, no loops needed
    def get_action_cowards(self):
        """ forward eta_w[all_archetypes, C, all_stages] by one day """
        new_eta_wc = np.zeros((len(self.sections), 1, self.n_stages))
        for i in range(len(self.sections)):
            new_eta_wc[i, 0, :] = np.asarray(self._get_action_cowards(i, self.eta_w[i, self.C, :]))

    def _get_action_cowards(self, idx, last_w):
        """
        :param idx:      The index of the section for which this function is called {0, ..., 15}
        :type idx:       int

        :param last_w:   The ratio of people (of this type) who went to work yesterday for each day
                         of infection
                         e.g. idx = 0, is the ratio (young primary coward sus W(t-1) / young primary cowards sus)
        :type last_w:    List[float]

        Params not needed here
        :i_stage:  Ratios for the day of infection this is (0 -> S, t_recovery -> R)
                    e.g. idx = 0, is the ratio (young primary coward sus / young primary cowards)
        :i_stage:   List[float]

        :return: ratio of people who will choose to work (indexed by i_stage)
        :rtype:  List[float]
        """
        w = []
        threshold_sw = self.coward_data['w-threshold']
        threshold_sh = self.coward_data['h-threshold']

        total_days = self.env_params['t-removal']
        unaware_days = self.env_params["t-symptoms"]

        ratio_over_threshold_w = 1 - ((threshold_sw - (1 - self.fluctuation / 2)) / self.fluctuation)
        ratio_over_threshold_h = 1 - ((threshold_sh - (1 - self.fluctuation / 2)) / self.fluctuation)

        w.append(last_w[0] * ratio_over_threshold_w + (1 - last_w[0]) * ratio_over_threshold_h)

        for i in range(total_days):
            if i < unaware_days:
                ratio_over_threshold_w = 1 - (
                        (threshold_sw - (1 - (i + 1) / unaware_days - self.fluctuation / 2)) / self.fluctuation)
                ratio_over_threshold_w = min(1.0, max(ratio_over_threshold_w, 0))
                ratio_over_threshold_h = 1 - (
                        (threshold_sh - (1 - (i + 1) / unaware_days - self.fluctuation / 2)) / self.fluctuation)
                ratio_over_threshold_h = min(1.0, max(ratio_over_threshold_h, 0))
            else:
                ratio_over_threshold_w = max((self.fluctuation / 2 - threshold_sw) / self.fluctuation, 0)
                ratio_over_threshold_h = max((self.fluctuation / 2 - threshold_sh) / self.fluctuation, 0)
            w.append(last_w[i] * ratio_over_threshold_w + (1 - last_w[i]) * ratio_over_threshold_h)

        ratio_over_threshold_w = 1 - ((threshold_sw - (1 - self.fluctuation / 2)) / self.fluctuation)
        ratio_over_threshold_h = 1 - ((threshold_sh - (1 - self.fluctuation / 2)) / self.fluctuation)

        w.append(last_w[total_days] * ratio_over_threshold_w + (1 - last_w[total_days]) * ratio_over_threshold_h)

        return w
