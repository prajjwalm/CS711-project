import argparse
import logging
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np

from constants import sections, s_pops, max_utility, job_risk, survival, env_params, player_data, player_types, T

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


def _add_args(parser: argparse.ArgumentParser):
    parser.add_argument("--eta-types", nargs=3, metavar=('nC', 'nP', 'nS'),
                        help="Ratios of the population types (will be scaled to ensure the sum to 1)"
                             " in case of a population simulation",
                        type=float, default=[0.33, 0.33, 0.34])
    pass


def _parse_args(args: argparse.Namespace):
    type_pops = np.asarray(args.eta_types)
    type_pops /= np.sum(type_pops)
    i_ratio = args.i_start / args.n_start
    return Population(type_pops=type_pops, i_ratio=i_ratio, t_max=args.t_max)


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

    # prototype enum
    C:              int = 0     # archetype index for type coward
    P:              int = 1     # archetype index for type planner
    S:              int = 2     # archetype index for type simple

    eta_iw:         np.ndarray
    eta_ih:         float

    T_max:          int

    # For graphs
    t_deaths_ot:    List[float]  # total deaths over time
    f_deaths_ot:    List[float]  # fresh deaths over time
    t_deaths_ot_s:  List[float]  # total deaths over time for each section
    t_utility_ot:   List[float]  # total utility over
    d_utility_ot:   List[float]  # daily utility over

    eta_w_ot:       List[float]  # Population going to work

    timeline:       np.ndarray  # time scale

    archtype_dict: Dict[int, str]      #Contains Labels for n_types

    # @formatter:on

    class ForcedExit(Exception):
        pass

    @staticmethod
    def _safe_divide(a: np.ndarray, b: np.ndarray):
        """ returns a/b if b != 0 else 0 """
        return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

    def __init__(self, type_pops: np.ndarray, i_ratio: float, t_max: int):
        self.coward_data = player_types["coward"]
        self.player_data = player_data
        self.p_h_delta = self.player_data['p-healthy-fluctuation']

        self.n_sections = len(sections)
        self.n_stages = env_params['t-removal'] + 1

        self.people = np.zeros((self.n_sections, self.n_types, self.n_stages))

        s_type_pops = np.expand_dims(np.array(s_pops), axis=1) * np.expand_dims(type_pops, axis=0)

        self.people[:, :, 1] = s_type_pops * i_ratio
        self.people[:, :, 0] = s_type_pops - self.people[:, :, 1]

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

        self.T_max = t_max

        self.t_deaths_ot = []
        self.f_deaths_ot = []
        self.t_deaths_ot_s = []

        self.t_utility_ot = []
        self.d_utility_ot = []

        self.eta_w_ot = []

        self.timeline = np.arange(self.T_max)

        self.archtype_dict = {0: 'Coward', 1: 'Planner', 2: 'Simple'}

        # TODO: set linear instead of step
        self.threshold_sw = np.where(job_risk < self.coward_data['job-risk-threshold'],
                                     self.coward_data['low-risk-w-threshold'],
                                     self.coward_data['w-threshold'])
        self.threshold_sh = np.where(job_risk < self.coward_data['job-risk-threshold'],
                                     self.coward_data['low-risk-h-threshold'],
                                     self.coward_data['h-threshold'])

    def _execute_infection(self):
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
        eta_w = self.eta_w[:, :, 0]
        eta_iw = np.expand_dims(self.eta_iw, axis=1)
        eta_i = eta_w * eta_iw + (1 - eta_w) * self.eta_ih

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

        self.t_deaths_ot_s.append(self.deaths)
        self.t_deaths_ot.append(np.sum(self.deaths, axis=0))
        self.f_deaths_ot.append(np.sum(fresh_deaths, axis=0))

        logger.debug("Fresh Deaths: " + str(np.sum(fresh_deaths, axis=0)))
        logger.debug("Total Deaths: " + str(np.sum(self.deaths, axis=0)))

        # 4. execute the eta_w transitions (note: eta_w will be wrt the new population distribution)
        eta_wi = self._safe_divide(eta_iw * eta_w, eta_i)  # P[W given they get I]
        eta_ws = self._safe_divide((1 - eta_iw) * eta_w, (1 - eta_i))  # P[W given they remain S]
        self.eta_w[:, :, -1] = 1 - self._safe_divide(fresh_recovered, self.people[:, :, -1])
        self.eta_w[:, :, 2:-1] = self.eta_w[:, :, 1:-2]
        self.eta_w[:, :, 1] = eta_wi
        self.eta_w[:, :, 0] = eta_ws

        # 5. execute X transitions (note: this is only for planner archetype)
        self.X[:, -1] = self._safe_divide(
            self.X[:, -1] * old_recovered[:, self.P] + self.X[:, -2] * fresh_recovered[:, self.P],
            self.people[:, self.P, -1]
        )
        self.X[:, 2:-1] = self.X[:, 1:-2]
        pop_infected_today = self._safe_divide(self.eta_w[:, self.P, 1], self.eta_w[:, self.P, 0])
        self.X[:, 1] = pop_infected_today * self.X[:, 0]
        self.X[:, 0] = (1 + pop_infected_today) * self.X[:, 0]
        self.X = np.clip(self.X, 0, 1 / 1 - self.h)

    def _update_utility(self):
        """ Takes the actions supplied, increments the utility and returns the risks of infection """
        i_loss = np.clip(1 - np.arange(self.n_stages) / env_params['t-infectious'], 0, 1)
        i_loss[-1] = 1
        i_loss = i_loss ** 2
        s_utility = np.einsum("ijk,ijk,i,k -> j", self.people, self.eta_w, max_utility, i_loss)
        logger.debug("Fresh utility: " + str(s_utility))
        self.net_utility += s_utility
        logger.debug("Total utility: " + str(self.net_utility))

        if self.t_utility_ot:
            self.t_utility_ot.append(s_utility+self.t_utility_ot[-1])
        else:
            self.t_utility_ot.append(s_utility)

        self.d_utility_ot.append(s_utility)


    def _get_action_coward(self):
        """
        Forwards eta_w[all_archetypes, C, all_stages] by one day. That is, using self parameters (in
        particular self.eta_w, the ratio of people in each cell of (section x archetype x i_stage)
        who had chosen to work yesterday (they might have been in a different cell per the last axis
        then)) set self.eta_w to the ratio of people in each cell who will choose to work today.
        """
        w = np.zeros((self.n_sections, self.n_stages,))

        ts = env_params["t-symptoms"]
        last_w = self.eta_w[:, self.C, :]

        ratio_over_threshold_w = 1 - ((np.expand_dims(self.threshold_sw, axis=1) - (
                1 - np.expand_dims(np.arange(ts), axis=0) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        ratio_over_threshold_h = 1 - ((np.expand_dims(self.threshold_sh, axis=1) - (
                1 - np.expand_dims(np.arange(ts), axis=0) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        # ratio_over_threshold_h = 1 - ((threshold_sh - (1 - np.arange(ts) / ts - self.p_h_delta / 2)) / self.p_h_delta)
        ratio_over_threshold_w = np.clip(ratio_over_threshold_w, 0, 1)
        ratio_over_threshold_h = np.clip(ratio_over_threshold_h, 0, 1)
        w[:, :ts] = last_w[:, :ts] * ratio_over_threshold_w + (1 - last_w[:, :ts]) * ratio_over_threshold_h

        w[:, ts:-1] = last_w[:, ts:-1] * np.maximum((self.p_h_delta / 2 - self.threshold_sw) / self.p_h_delta, 0) \
                      + (1 - last_w[:, ts:-1]) * np.maximum((self.p_h_delta / 2 - self.threshold_sh) / self.p_h_delta,
                                                            0)
        w[:, -1] = 1

        self.eta_w[:, self.C, :] = np.asarray(w)

    def _get_action_simple(self):
        u_iw = (self.eta_iw - self.eta_ih) * (1 - survival) * self.player_data["u-death"]
        p_max = 1 + self.p_h_delta / 2 - np.arange(self.n_stages) / env_params["t-symptoms"]
        cutoff = np.expand_dims(np.sqrt(u_iw / max_utility), axis=1)
        eta_w = np.where(cutoff > 1, 0, (np.expand_dims(p_max, axis=0) - cutoff) / self.p_h_delta)
        eta_w[:, -1] = 1
        self.eta_w[:, self.S, :] = np.clip(eta_w, 0, 1)

    def _get_action_planner(self):
        self.X = self.eta_w[:, self.P, :] + self.h * self.X

        eta_w_s = self.eta_w[:, self.S, :]
        eta_w_del = np.minimum(1 - eta_w_s, eta_w_s) * 0.4
        eta_w_min = eta_w_s - eta_w_del
        eta_w_max = eta_w_s + eta_w_del

        self.eta_w[:, self.P, :] = (eta_w_min - eta_w_max) * (1 - self.h) * self.X + eta_w_max

    def simulate(self):
        try:
            for T[0] in range(self.T_max):
                logger.info("Population sections:\n" + str(self.people))
                logger.info("Percentage working:\n" + str(self.eta_w))
                logger.info("Final Utility: {}".format(self.net_utility + self.deaths * player_data['u-death']))

                self._get_action_coward()
                self._get_action_simple()
                # simple must be called before planner !
                self._get_action_planner()
                self._update_utility()
                self._execute_infection()
        except self.ForcedExit as e:
            logger.info(e)
            print(e)

        logger.info("Population sections:\n" + str(self.people))
        logger.info("Percentage working:\n" + str(self.eta_w))
        logger.info("Final Utility: {}".format(self.net_utility + self.deaths * player_data['u-death']))

    def total_death_plot(self):
        self.t_deaths_ot = np.asarray(self.t_deaths_ot)

        for i in range(self.n_types):
            plt.plot(self.timeline, self.t_deaths_ot[:,i], label=self.archtype_dict[i])

        plt.xlabel("Time")
        plt.ylabel("Death percentage")
        plt.grid()
        plt.legend()

        plt.savefig("graphs/total_deths.jpg")
#        plt.show()

    def fresh_death_plot(self):
        self.f_deaths_ot = np.asarray(self.f_deaths_ot)

        for i in range(self.n_types):
            plt.plot(self.timeline, self.f_deaths_ot[:,i], label=self.archtype_dict[i])


        plt.xlabel("Time")
        plt.ylabel("Daily Death percentage")
        plt.grid()
        plt.legend()
        plt.savefig("graphs/fresh_deaths.jpg")
#        plt.show()

    def total_utility_plot(self):
        self.t_utility_ot = np.asarray(self.t_utility_ot)


        for i in range(self.n_types):
            plt.plot(self.timeline, self.t_utility_ot[:,i], label=self.archtype_dict[i])


        plt.xlabel("Time")
        plt.ylabel("Total Utility")
        plt.grid()
        plt.legend()
        plt.savefig("graphs/total_utility.jpg")
#        plt.show()

    def daily_utility_plot(self):
        self.d_utility_ot = np.asarray(self.d_utility_ot)


        for i in range(self.n_types):
            plt.plot(self.timeline, self.d_utility_ot[:,i], label=self.archtype_dict[i])


        plt.xlabel("Time")
        plt.ylabel("Total Utility")
        plt.grid()
        plt.legend()
        plt.savefig("graphs/daily_utility.jpg")
#        plt.show()

    def total_death_plot_sections(self):
        self.t_deaths_ot_s = np.asarray(self.t_deaths_ot_s)
        print(self.t_deaths_ot_s[-1])
        for i in range(self.n_types):
            for j in range(self.n_sections):
                plt.plot(self.timeline, self.t_deaths_ot_s[:,j,i], label=str(i)+"+"+str(j))


        plt.xlabel("Time")
        plt.ylabel("Death percentage")
        plt.grid()
        plt.legend()
        plt.savefig("graphs/total_deaths_per_section.jpg")
#        plt.show()

    def plot_graphs(self):
        self.total_death_plot()
        self.fresh_death_plot()
        self.total_utility_plot()
        self.daily_utility_plot()
        self.total_death_plot_sections()
