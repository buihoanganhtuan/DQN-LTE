""" # main body of the simulation
import settings
import math
import numpy as np
from devices import devices

settings.init()

mode = "episodic"
num_ep = 12000
num_step = 100000
n = 7e2
R = 54
W_RAR = 5
N_RAR = 12
T_RAO = 5
T = 100e-3
sf_len = 1e-3
T_norm = math.ceil(T/T_RAO/sf_len)
Npt_max = 10
m3_harq_p = m4_harq_p = 0.0
m3_harq_max = m4_harq_max = 5
current_sf = 0
p_barring = 1
a_space = np.arange(0, 1.05, 0.05)
n_actions = a_space.shape[0]
dev = devices(n=n, R=R, W_RAR=W_RAR, N_RAR=N_RAR, T_RAO=T_RAO, T=T, Npt_max=Npt_max, m3_harq_max=m3_harq_max, m4_harq_max=m4_harq_max, m3_harq_p=m3_harq_p, m4_harq_p=m4_harq_p)
nextEvent = settings.scheduler.getNextEvent()
current_sf = nextEvent
previousEvent = nextEvent

if mode == "episodic":
    ep_count = 0
    while ep_count < num_ep:
        while nextEvent != -1:
            dev.sendPreamble(current_sf, p_barring=p_barring)
            dev.sendMsg3(current_sf)
            dev.receiveMsg4(current_sf)
            p_barring = a_space[np.random.randint(0, n_actions)]
            nextEvent = settings.scheduler.getNextEvent()
            current_sf = nextEvent
        ep_count += 1
        print(dev.getBlock())
        settings.scheduler.clear()
        dev.refresh(mode, current_sf)
        nextEvent = settings.scheduler.getNextEvent()
        current_sf = nextEvent
elif mode == "cont":
    while (current_sf//T_RAO) < num_step:
        if (nextEvent//T_norm) > (previousEvent//T_norm) | (nextEvent==-1):
            dev.refresh(mode, current_sf) """
        
