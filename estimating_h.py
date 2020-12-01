import numpy as np

target_days = 4

u_economic_w = 1.2
u_death = 20000
death_prob = 4/90



h = 0.4

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
probab = 1 - (1-h)*x
print(probab)

# expected ratio 1.133 for h = 0.5

#last_weeks_actions = ["W","H","H","W","W","H","W"]
#last_weeks_actions = np.array(last_weeks_actions) == "W"
#print(last_weeks_actions)


