import numpy as np

target_days = 4


u_economic_w = 1.2
u_death = 20000
death_prob = 1 - 0.95644

coeff_i = 0.05
coeff_wi =  0.2

total_infectious = 0.04991064123654763
working_infectious = 0.005820863793475713

p_h_max = 23/24

eta_ih = total_infectious * coeff_i
eta_iw = np.clip(1 - (1 - total_infectious * coeff_i) * (1 - working_infectious * coeff_wi), eta_ih, 1)

cutoff_p_h = np.sqrt(u_death * death_prob * (eta_iw - eta_ih) / u_economic_w)


_eta_w = 0.16055732
assert -0.01 < (p_h_max - np.sqrt(u_death * death_prob * (eta_iw - eta_ih) / u_economic_w)) / 0.25 - _eta_w < 0.01

h = 1 - 1/(2*((p_h_max - np.sqrt(u_death * death_prob * (eta_iw - eta_ih) / u_economic_w)) / 0.25))


weights = np.logspace(0,6,num=7,base = h)

loops = 10000
x = 0

for _ in range(loops):
    perm = np.arange(7)
    perm = np.random.permutation(perm)
    perm = perm[:target_days]

    arr = np.zeros(7)
    for idx in perm:
        arr[idx] = 1

    x += np.sum(arr*weights)

x /= loops
p = 1 - (1-h)*x
print(p)

#
# WHWHWHWHWHWHWHWH(50%W).
# p = f(h) for a certain work history X p = 1/2(1-h)
# p = eta_w
# eta_w = g(params)
# h = f-1(g(params)) = 1 - 1/(2*eta_w[Raghav])
#
