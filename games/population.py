import logging
from typing import List, Dict

import numpy as np

from constants import sections, s_pops, max_utility, job_risk, survival, env_params, player_data, player_types
from players import Planner

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
    net_utility:    float
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

        self.people[:, self.S, 0] = np.array(s_pops) * 0.99
        self.people[:, self.S, 1] = np.array(s_pops) - self.people[:, self.S, 0]

        self.eta_w = np.zeros((self.n_sections, 3, self.n_stages))
        self.eta_w[:, self.S, 0] = 1
        self.eta_w[:, self.S, -1] = 1

        self.net_utility = 0
        self.deaths = np.zeros((self.n_sections, self.n_types))

        self.eta_iw = np.zeros(self.n_sections)
        self.eta_ih = 0

        self.X = np.zeros((self.n_sections, self.n_stages))
        h = max_utility / (max_utility + (1 - survival) * self.player_data["u-death"])

        self.X[:, 0] = 1 / (1 - h)

    def execute_infection(self):
        """
        Adjusts self.people and self.eta_w forward by one day wrt the pandemic.

        While the adjustment of self.people is trivial, an example would help to understand how
        self.eta_w is modified: say, get_action_cowards sets self.eta_w[1, 0, 3] = 0.7, (ie. we
        compute 70% of young primary cowards at their 3rd day of infection choose work) then, this
        function will set self.eta_w[1, 0, 4] = 0.7 (so that when get_action_cowards is called
        again, they see that 70% of people currently in their 4th day of infection had chosen work
        yesterday)
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

        self.people[:, :, 2:-1] = self.people[:, :, 1:-2]
        self.people[:, :, -1] += fresh_recovered
        self.people[:, :, 0] -= fresh_infected
        self.people[:, :, 1] = fresh_infected
        self.deaths += fresh_deaths

        # 4. execute the eta_w transitions (note: eta_w will be wrt the new population distribution)
        eta_wi = self.safe_divide(eta_iw * eta_w, eta_i)  # P[W given they get I]
        eta_ws = self.safe_divide((1 - eta_iw) * eta_w, (1 - eta_i))  # P[W given they remain S]
        self.eta_w[:, :, 2:-1] = self.eta_w[:, :, 1:-2]
        self.eta_w[:, :, -1] = 1 - self.safe_divide(fresh_recovered, self.people[:, :, -1])
        self.eta_w[:, :, 1] = eta_wi
        self.eta_w[:, :, 0] = eta_ws

    def update_utility(self):
        """ Takes the actions supplied, increments the utility and returns the risks of infection """
        i_loss = np.clip(1 - np.arange(self.n_stages) / env_params['t-infectious'], 0, 1)
        i_loss[-1] = 1
        i_loss = i_loss ** 2
        s_utility = np.einsum("ijk,ijk,i,k -> j", self.people, self.eta_w, max_utility, i_loss)
        self.net_utility += np.sum(s_utility)

    def get_action_cowards(self):
        """
        Forwards eta_w[all_archetypes, C, all_stages] by one day. That is, using self parameters (in
        particular self.eta_w, the ratio of people in each cell of (section x archetype x i_stage)
        who had chosen to work yesterday (they might have been in a different cell per the last axis
        then)) set self.eta_w to the ratio of people in each cell who will choose to work today.
        """
        new_eta_wc = np.zeros((self.n_sections, 1, self.n_stages))

        # TODO: @srajit, @ravi remove this loop too
        for i in range(self.n_sections):
            new_eta_wc[i, 0, :] = np.asarray(self._get_action_cowards(i, self.eta_w[i, self.C, :]))
        self.eta_w[:, [self.C], :] = new_eta_wc

    def _get_action_cowards(self, idx: int, last_w: np.ndarray):
        """
        :param idx:      The index of the section for which this function is called {0, ..., 15}
        :param last_w:   The ratio of people (of this type) who went to work yesterday for each day
                         of infection

        :return: ratio of people who will choose to work (indexed by i_stage)
        :rtype:  List[float]
        """
        w = []
        threshold_sw = self.coward_data['w-threshold']
        threshold_sh = self.coward_data['h-threshold']

        ts = env_params["t-symptoms"]
        ##########################################################################################################

        ratio_over_threshold_w = 1 - (
                    (threshold_sw - (1 - np.arange(ts - 1) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        ratio_over_threshold_h = 1 - (
                    (threshold_sh - (1 - np.arange(ts - 1) / ts - self.p_h_delta / 2)) / self.p_h_delta)

        # TODO: @srajit, @ravi, you'll probably get a list of two numpy arrays, this is not what you intended, to make
        #  this work, set w to be a numpy array and adjust accordingly
        w.append(last_w[:ts - 1] * ratio_over_threshold_w + (1 - last_w[:ts - 1]) * ratio_over_threshold_h)

        w.append(last_w[ts:] * max((self.p_h_delta / 2 - threshold_sw) / self.p_h_delta, 0) \
                 + (1 - last_w[ts:]) * max((self.p_h_delta / 2 - threshold_sh) / self.p_h_delta, 0))
        # TODO: @srajit, @ravi are you sure this works for last column (for recovered)

        assert len(w) == self.n_stages

        return w

    def get_action_simple(self):
        # for i_stage = 0, c1 * U^2 - c2 > 0 implies person going to work
        # where U is a uniform R.V.
        # c1 = u_per_capita and c2 = (eta_iw - eta_ih) * ( 1 - survival) * death_util
        # for i_stage = 21, Choose W
        # for i_stage = {1,..,20} , c1 is same, c2 is same
        # eta_w = u_max - sqrt(c2/c1) / u_max - u_min
        # u_max, u_min = (1 - i/t-symptoms) +- fluctuation/2

        c2 = (self.eta_iw - self.eta_ih) * (1 - survival) * self.player_data["u-death"]
        assert c2.shape == (self.n_sections,)
        logger.debug("eta_iw - eta_ih: \n" + str(self.eta_iw - self.eta_ih))
        logger.debug("survival: \n" + str(survival))
        logger.debug("c2: \n" + str(c2))

        c1 = max_utility
        assert c1.shape == (self.n_sections,)
        logger.debug("u_per_capita: \n" + str(c1))

        p_max = 1 - np.arange(self.n_stages) / env_params["t-symptoms"] + self.p_h_delta / 2
        p_max[-1] = 1
        assert p_max.shape == (self.n_stages,)
        logger.debug("health belief max: \n" + str(p_max))

        eta_w = (np.expand_dims(p_max, axis=0) - np.expand_dims(np.sqrt(c2 / c1), axis=1)) / self.p_h_delta
        eta_w = np.where(np.expand_dims(np.sqrt(c2 / c1), axis=1) > 1, 0, eta_w)

        eta_w[:, -1] = 1
        assert eta_w.shape == (self.n_sections, self.n_stages)
        eta_w = np.clip(eta_w, 0, 1)
        logger.debug("eta_w: \n" + str(eta_w))

        self.eta_w[:, self.S, :] = eta_w

    def get_action_planner(self):
        #
        # P[W_it = 1] = m(i-stage_i, delta-i_t).X_it + c(i-stage_i, delta-i_t)
        # n(W_t) = m(i-stage_i, delta-i_t).X_t + c(i-stage_i, delta-i_t)        , for each cell
        # and, X_t = n(W_{t-1}) + h(params)X_{t-1}
        #
        # so we track X_t and n(W_t) for all t
        #

        # step 0: find h, delta_i
        delta_i = self.eta_iw - self.eta_ih

        assert self.h.shape == delta_i.shape == (self.n_sections,)

        p_max = Planner.max_eta_w(self.h, delta_i)
        p_min = Planner.min_eta_w(self.h, delta_i)

        assert p_max.shape == p_min.shape == (self.n_sections, self.n_stages)

        eta_w = self.eta_w[:, self.P, :]

        # TODO: adjust self.X with population shifts in execute_infection
        self.X = eta_w + self.h * self.X

        # p_max corresponds to self.X = 0       [15,22]
        # p_min     '       '  self.X = /(1-h) [15,22]

        eta_w = (p_min - p_max) * (1 - self.h) * self.X + p_max
        self.eta_w[:, self.P, :] = eta_w

    def simulate(self):
        try:
            for t in range(env_params['t-max']):
                self.get_action_simple()
                self.update_utility()
                self.execute_infection()
                logger.info("Population sections:\n" + str(self.people))
                logger.info("Percentage working:\n" + str(self.eta_w))
                logger.info("net_utility: {:.2f}".format(self.net_utility))
        except self.ForcedExit as e:
            logger.info(e)
            print(e)
