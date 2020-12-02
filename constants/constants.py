from json import load

import numpy as np

# TODO: setup logging here
print("CONSTANTS CODE RUN")

with open("data.json") as f:
    _raw_data = load(f)

# Environment parameters
env_params = _raw_data['game-params']
n_stages = env_params['t-removal'] + 1

# Player and archetype parameters
player_data = _raw_data['base-player']
player_types = _raw_data['player-types']

# Section parameters and demographics
sections = []
s_pops = []
_s_params = []
_params_order = _raw_data['jobs-param-order'] + ["danger"]

for k1, v1 in _raw_data["jobs"].items():
    for k2, v2 in _raw_data["population"].items():
        sections.append(k2 + " " + k1)
        _s_params.append(v1 + [v2['danger']])
        s_pops.append(v2['ratio'] * v2['job-dist'][k1])

assert _params_order[1] == "job-importance" and _params_order[2] == "economic-status"
max_utility = \
    player_data['x-job-importance'] * np.asarray([x[1] for x in _s_params]) + \
    player_data['x-economic-status'] * (1 - np.asarray([x[2] for x in _s_params]))

assert _params_order[0] == "job-risk"
job_risk = np.asarray([x[0] for x in _s_params])

assert _params_order[3] == "danger"
survival = 1 - np.asarray([0.1 * x[3] ** 2 for x in _s_params])

assert max_utility.shape == survival.shape == job_risk.shape == (len(sections),)
