import logging
from typing import List
import numpy as np
from .base_player import BasePlayer

logger: logging.Logger


def _init():
    """ all file related initialization code """
    global logger
    logger = logging.getLogger("Log")


class Coward(BasePlayer):
    last_action: str  # Logs last action
    caution_multiplier: float

    def __init__(self, env, *args, **kwargs):
        super().__init__(env, *args, **kwargs)
        self.last_action = "W"
        self.caution_multiplier = 1.0

    def plan(self):
        """
        Strategy:
        Doesn't care about utilities. Doesn't plan into the future. Works if
        feeling healthy today, else stays at home.
        [TODO: affect threshold with current number of cases?]
        """

        """
        New additions Coward_2.0
        Job risk pe dependency add karni hai
        Infection risk bohot kam hai to work karna chahiye
        """

        assert len(self.action_plan) == 0

        """
        Variables available:
        u_economic_w -> (u_economic_max)
        infection_risk :::->:
        self.w_infection_risk
        self.h_infection_risk
        """
        # Srajit blink twice if you are being held captive

        # Lodu, agar high hua risk to 1 hojayega threshold
        # 0 risk pe bhi threshold 1
        job_risk_threshold = 0.01       # value pm se cross check karni hai
        lower_threshold = 0.9
        upper_threshold = 0.96

        if self.w_infection_risk < job_risk_threshold:
            lower_threshold = 0.420
            upper_threshold = 0.69

        lower_threshold *= self.caution_multiplier
        upper_threshold *= self.caution_multiplier

        # lower_threshold = np.clip(self.w_infection_risk * 9, 0 , 1)
        # upper_threshold = np.clip(self.w_infection_risk * 10, 0 , 1)

        threshold = lower_threshold if self.last_action == "H" else upper_threshold

        action = "W" if self.p_healthy > threshold else "H"
        self.action_plan.append(action)

        self.last_action = self.action_plan[-1]

        logger.debug("Health belief is {0:.2f}, so choosing {1}".format(
                self.p_healthy, self.action_plan[-1]
        ))
    """
    Everything should be made as simple as possible, but no simpler
                                                - Albert Einstein
    Everything should be made as simple as possible
                                                - Srajit Kumar
    Everything should be made as difficult as possible
                                                - Prajjwal Mishra
    Everything should just be (Ye raaz bhi usi ke saath chala gya)
                                                - Shreyash Ravi
    Everything
                                                -Raghav Maheshwari
    """
    def update(self, actions: List[str], self_idx: int):
        work_people = actions.count("W")
        total_people = len(actions)
        attendance = work_people / total_people

        safe_attendance_limit = 0.5
        if attendance > safe_attendance_limit:
            self.caution_multiplier = 1000000
        else:
            self.caution_multiplier = 1
