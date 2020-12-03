import logging
from typing import List, Dict

import numpy as np

from constants import sections, s_pops, max_utility, job_risk, survival, env_params, player_data, player_types

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Population:
    # @formatter:off

    # size range over which a persons belief of his own health (type) varies
    p_h_delta:    float

    # constants used in various archetypes
    coward_data:    Dict[str, float]
    player_data:    Dict[str, float]

    # player division across ( section x archetype x i_stage )
    n_sections:     int
    n_stages:       int         # number of stages
    n_types:        int = 3     # number of archetypes
    people:         np.ndarray  # ratio of population for each cell
    deaths:         np.ndarray  # ( section x archetype )
    eta_w:          np.ndarray  # ratio of people who worked / will work in that cell

    # for planner
    X:              np.ndarray                              # work measure, (section x i_stage)
    h:              float = player_types['planner']['h']    # memory over past days
    influence_cap:  float = player_types['planner']['cap']  # max influence on P[W/H] due to X

    # utility measures
    net_utility:    np.ndarray
        # TODO: calibrate the above (consts and formulae) with base_player

    # prototype enum
    C:              int = 0     # archetype index for type coward
    P:              int = 1     # archetype index for type planner
    S:              int = 2     # archetype index for type simple

    eta_iw:         np.ndarray
    eta_ih:         float

    # @formatter:on

    class ForcedExit(Exception):
        pass

    @staticmethod
    def safe_divide(a: np.ndarray, b: np.ndarray):
        """ returns a/b if b != 0 else 0 """
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def __init__(self):
        self.coward_data = player_types["coward"]
        self.player_data = player_data
        self.p_h_delta = self.player_data['p-healthy-fluctuation']

        self.n_sections = len(sections)
        self.n_stages = env_params['t-removal'] + 1

        self.people = np.zeros((self.n_sections, self.n_types, self.n_stages))

        section_type_pops = np.expand_dims(np.array(s_pops), axis=1) * np.expand_dims(np.asarray([0.33, 0.33, 0.34]), axis=0)
        self.people[:, :, 0] = section_type_pops * 0.99
        self.people[:, :, 1] = section_type_pops - self.people[:, :, 0]

        self.eta_w = np.zeros((self.n_sections, 3, self.n_stages))
        self.eta_w[:, :, 0] = 1
        self.eta_w[:, :, -1] = 1

        self.net_utility = np.zeros(self.n_types)
        self.deaths = np.zeros((self.n_sections, self.n_types))

        self.eta_iw = np.zeros(self.n_sections)
        self.eta_ih = 0

        self.X = np.zeros((self.n_sections, self.n_stages))
        h = max_utility / (max_utility + (1 - survival) * self.player_data["u-death"])

        self.X[:, 0] = 1 / (1 - h)

    def execute_infection(self):
        """
        Adjusts self.people, self.eta_w and self.X forward by one day wrt the pandemic.
        """

        # 1. Find the probability of infection for S work and S home, ie. eta_iw, eta_ih respectively
        infectious_mask = np.zeros(self.n_stages)
        infectious_mask[env_params['t-infectious']:-1] = 1
        s_working_infectious = np.einsum("ijk,ijk,k -> j", self.people, self.eta_w, infectious_mask)

        working_infectious = np.sum(s_working_infectious)  # ratio of (working, infected) / total population
        infected = np.sum(self.people, axis=(0, 1))
        assert infected.shape == (self.n_stages,)
        total_infectious = np.sum(infected[env_params['t-infectious']:-1])
        total_infected = np.sum(infected[1:-1])

        coeff_i = 0.05
        coeff_wi = 1 * job_risk

        self.eta_ih = total_infectious * coeff_i
        self.eta_iw = np.clip(1 - (1 - total_infectious * coeff_i) * (1 - working_infectious * coeff_wi), self.eta_ih,
                              1)
        logger.debug("total-infectious: {0}, working-infectious : {1}".format(total_infectious, working_infectious))
        logger.debug("eta_iw: {0}, eta_ih: {1:.2f}".format(str(self.eta_iw), self.eta_ih))

        # 2. find out the ratio of susceptible that are getting infected
        eta_w = self.eta_w[:, :, 0]  # ratio of people who worked (section x archetype) among S
        eta_iw = np.expand_dims(self.eta_iw, axis=1)
        eta_i = eta_w * eta_iw + (1 - eta_w) * self.eta_ih
        # ratio of people who got infected (section x archetype)
        assert eta_i.shape == (self.n_sections, self.n_types)

        if total_infected == 0:
            raise self.ForcedExit("Infection Eradicated")

        # 3. execute the population transitions
        fresh_infected = eta_i * self.people[:, :, 0]
        fresh_recovered = self.people[:, :, -2] * np.expand_dims(survival, axis=1)
        fresh_deaths = self.people[:, :, -2] - fresh_recovered
        old_recovered = self.people[:, :, -1]

        self.people[:, :, 2:-1] = self.people[:, :, 1:-2]
        self.people[:, :, -1] += fresh_recovered
        self.people[:, :, 0] -= fresh_infected
        self.people[:, :, 1] = fresh_infected
        self.deaths += fresh_deaths

        logger.debug("Fresh Deaths: " + str(np.sum(fresh_deaths, axis=0)))
        logger.debug("Total Deaths: " + str(np.sum(self.deaths, axis=0)))

        # 4. execute the eta_w transitions (note: eta_w will be wrt the new population distribution)
        eta_wi = self.safe_divide(eta_iw * eta_w, eta_i)  # P[W given they get I]
        eta_ws = self.safe_divide((1 - eta_iw) * eta_w, (1 - eta_i))  # P[W given they remain S]
        self.eta_w[:, :, -1] = 1 - self.safe_divide(fresh_recovered, self.people[:, :, -1])
        self.eta_w[:, :, 2:-1] = self.eta_w[:, :, 1:-2]
        self.eta_w[:, :, 1] = eta_wi
        self.eta_w[:, :, 0] = eta_ws

        # 5. execute X transitions (note: this is only for planner archetype)
        self.X[:, -1] = self.safe_divide(
            self.X[:, -1] * old_recovered[:, self.P] + self.X[:, -2] * fresh_recovered[:, self.P],
            self.people[:, self.P, -1]
        )
        self.X[:, 2:-1] = self.X[:, 1:-2]
        self.X[:, 1] = self.X[:, 0]

    def update_utility(self):
        """ Takes the actions supplied, increments the utility and returns the risks of infection """
        i_loss = np.clip(1 - np.arange(self.n_stages) / env_params['t-infectious'], 0, 1)
        i_loss[-1] = 1
        i_loss = i_loss ** 2
        s_utility = np.einsum("ijk,ijk,i,k -> j", self.people, self.eta_w, max_utility, i_loss)
        logger.debug("Fresh utility: " + str(s_utility))
        self.net_utility += s_utility
        logger.debug("Total utility: " + str(self.net_utility))

    def get_action_cowards(self):
        """
        Forwards eta_w[all_archetypes, C, all_stages] by one day. That is, using self parameters (in
        particular self.eta_w, the ratio of people in each cell of (section x archetype x i_stage)
        who had chosen to work yesterday (they might have been in a different cell per the last axis
        then)) set self.eta_w to the ratio of people in each cell who will choose to work today.
        """
        w = np.zeros((self.n_sections, self.n_stages,))
        threshold_sw = np.zeros((self.n_sections,))
        threshold_sh = np.zeros((self.n_sections,))
        job_risk_threshold = self.coward_data['job-risk-threshold']

        threshold_sw = np.where(self.eta_iw < job_risk_threshold, self.coward_data['low-risk-w-threshold'], self.coward_data['w-threshold'])
        threshold_sh = np.where(self.eta_iw < job_risk_threshold, self.coward_data['low-risk-h-threshold'], self.coward_data['h-threshold'])

        ts = env_params["t-symptoms"]
        last_w = self.eta_w[:, self.C, :]

        ratio_over_threshold_w = 1 - ((np.expand_dims(threshold_sw, axis = 1) - (1 - np.expand_dims(np.arange(ts),axis = 0) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        ratio_over_threshold_h = 1 - ((np.expand_dims(threshold_sh, axis = 1) - (1 - np.expand_dims(np.arange(ts),axis = 0) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        # ratio_over_threshold_h = 1 - ((threshold_sh - (1 - np.arange(ts) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        ratio_over_threshold_w = np.clip(ratio_over_threshold_w, 0, 1)
        ratio_over_threshold_h = np.clip(ratio_over_threshold_h, 0, 1)
        w[:, :ts] = last_w[:, :ts] * ratio_over_threshold_w + (1 - last_w[:, :ts]) * ratio_over_threshold_h

        w[:, ts:-1] = last_w[:, ts:-1] * np.maximum((self.p_h_delta / 2 - threshold_sw) / self.p_h_delta, 0) \
                + (1 - last_w[:, ts:-1]) * np.maximum((self.p_h_delta / 2 - threshold_sh) / self.p_h_delta, 0)
        w[:, -1] = 1

        self.eta_w[:, self.C, :] = np.asarray(w)

    def get_action_simple(self):
        self.eta_w[:, self.S, :] = self._get_action_simple()

    def _get_action_simple(self):
        u_iw = (self.eta_iw - self.eta_ih) * (1 - survival) * self.player_data["u-death"]
        p_max = 1 + self.p_h_delta / 2 - np.arange(self.n_stages) / env_params["t-symptoms"]
        cutoff = np.expand_dims(np.sqrt(u_iw / max_utility), axis=1)
        eta_w = np.where(cutoff > 1, 0, (np.expand_dims(p_max, axis=0) - cutoff) / self.p_h_delta)
        eta_w[:, -1] = 1
        return np.clip(eta_w, 0, 1)

    def get_action_planner(self):
        self.X = self.eta_w[:, self.P, :] + self.h * self.X

        eta_w_s = self._get_action_simple()
        eta_w_del = np.minimum(1 - eta_w_s, eta_w_s) * 0.4
        eta_w_min = eta_w_s - eta_w_del
        eta_w_max = eta_w_s + eta_w_del

        self.eta_w[:, self.P, :] = (eta_w_min - eta_w_max) * (1 - self.h) * self.X + eta_w_max

    def simulate(self):
        try:
            for t in range(env_params['t-max']):
                self.get_action_cowards()
                self.get_action_simple()
                self.get_action_planner()
                self.update_utility()
                self.execute_infection()
                logger.info("Population sections:\n" + str(self.people))
                logger.info("Percentage working:\n" + str(self.eta_w))
                logger.info("Final Utility: {}".format(self.net_utility + self.deaths * player_data['u-death']))
        except self.ForcedExit as e:
            logger.info(e)
            print(e)
