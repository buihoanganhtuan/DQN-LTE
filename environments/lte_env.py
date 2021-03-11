from rl2.core import Env, Space
from rl2.specialized_devices.ltedevices import ACB_Devices
from rl2.lte_util import EventScheduler
import random
import numpy as np

class ACB_env(Env):
    def __init__(self, *args, **kwargs):
        assert 'mode' in kwargs, 'You must input a mode of operation of the environmment ("episodic" or "cont")'
        assert 'control_backoff' in kwargs, 'You must decide whether or not the backoff indicator is fixed or not'
        assert 'control_tbar' in kwargs, 'You must decide whether or not the mean barring time is fixed or not'
        self.mode = kwargs['mode']
        self.control_backoff = kwargs['control_backoff']
        self.control_tbar = kwargs['control_tbar']
        assert not (self.control_backoff & self.control_tbar), 'The agent should not be controlling both T_bar and BI'
        self.config = kwargs
        self.devices = ACB_Devices(**kwargs)
        self.event_scheduler = EventScheduler()
        self.event_scheduler.addEvent(self.devices.act_time.tolist())
        self.current_sf = 0
        self.start_offset = kwargs.get('start_offset', 0)
        self.end_trail = kwargs.get('end_trail', 0)
        self.T_RAO = self.config['T_RAO']
        if not self.control_backoff:
            if self.control_tbar:
                #a = np.array([0, 10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 960, 1920]) # mean barring time
                a = np.array([0, 50, 100, 200, 400, 800, 1200, 1600, 2000, 2400, 2800, 3200, 3600, 4000]) # mean barring time
                #a = np.array([0, 50, 100, 200, 400, 800, 1600, 3200]) # mean barring time for 1st time UE
                #a2 = np.array([0, 50, 100, 200, 400, 800, 1600, 3200]) # mean barring time for 2nd time UE
                b = np.arange(0.05, 1+0.05, 0.05)
                #b = np.arange(0.1, 1+0.1, 0.1)
                #b = np.arange(0.3, 1+0.1, 0.1)
                b = b[: :-1]
                self.action_space = LTE_action_space(np.array([[c,d] for c in b for d in a]))
                #self.action_space = LTE_action_space(np.array([[c,d,e] for c in b for d in a for e in a2]))
            else:
                self.action_space = LTE_action_space(np.arange(0., 1.+0.01, 0.01))
        else:
            a = np.array([0, 10, 20, 30, 40, 60, 80, 120, 160, 240, 320, 480, 960]) # backoff window length
            b = np.arange(0.05, 1+0.05, 0.05)
            b = b[: :-1]
            self.action_space = LTE_action_space(np.array([[c,d] for c in b for d in a]))
        #self.recent_action = self.action_space.actions[0]
        self.recent_action = 0

    def step(self, action_index):
        assert self.action_space.contains(action_index), 'action index must be between 0 and {}, got {}'.format(self.action_space.nb_actions - 1, action_index)
        # Let the environment interprete the action
        action = self.action_space.actions[action_index]
        self.current_sf = self.event_scheduler.getNextEvent()
        # If the subframe of the next event is not a RAO, then execute
        # the events until we reach a RAO
        while True:
            events = self.devices.sendMsg3(self.current_sf)
            events += self.devices.receiveMsg4(self.current_sf)
            self.event_scheduler.addEvent(events)
            if self.current_sf % self.T_RAO == 0:
                break
            self.current_sf = self.event_scheduler.getNextEvent()

        rec, done, info, e = self.devices.sendPreamble(self.current_sf, control_param=action) if not self.control_backoff else \
                                self.devices.sendPreamble(self.current_sf, control_param=action[0], resched_col_param=action[1], resched_fail_param=action[1])
        events += e
        #observation = [*list(rec.values()), self.recent_action] if self.action_space.action_dims == 0 else [*list(rec.values()), *self.recent_action]
        self.recent_action = action_index
        observation = [*list(rec.values()), self.recent_action]
        if self.action_space.action_dims == 1:
            info['p_bar'] = action
        elif self.action_space.action_dims == 2:
            if self.control_backoff:
                info['p_bar'] = action[0]
                info['bo'] = action[1] / self.action_space.max_actions[1]
            else:
                info['p_bar'] = action[0]
                info['t_bar'] = action[1] / self.action_space.max_actions[1]
        elif self.action_space.action_dims == 3:
            info['p_bar'] = action[0]
            info['t_bar_0'] = action[1] / self.action_space.max_actions[1]
            info['t_bar_1'] = action[2] / self.action_space.max_actions[2]
        else:
            raise NotImplementedError('action space dimension not supported')

        reward = list(rec.values())

        #self.recent_action = action
        self.event_scheduler.addEvent(events)
        if done:
            if self.mode == 'episodic':
                while self.current_sf != -1:
                    events = self.devices.sendMsg3(self.current_sf)
                    events += self.devices.receiveMsg4(self.current_sf)
                    self.event_scheduler.addEvent(events)
                    self.current_sf = self.event_scheduler.getNextEvent()
    
        return observation, reward, done, info

    def reset(self):
        rec, events = self.devices.refresh(self.mode, self.current_sf)
        self.event_scheduler.addEvent(events)
        #self.recent_action = self.action_space.actions[0]
        self.recent_action = 0
        #observation = [*list(rec.values()), self.recent_action] if self.action_space.action_dims == 0 else [*list(rec.values()), *self.recent_action]
        observation = [*list(rec.values()), self.recent_action]
        return observation

class LTE_action_space(Space):
    def __init__(self, actions=None):
        assert actions is not None, 'Your environment has no action space'
        assert isinstance(actions, np.ndarray), 'Action space must be a numpy array. Got {} instead'.format(type(actions))
        self.actions = actions
        self.nb_actions = len(actions)
        self.max_actions = np.max(actions, axis=0)
        self.action_dims = len(actions[0])

    def sample(self, seed=None):
        return self.actions[random.sample(range(self.nb_actions), 1)]

    def contains(self, x):
        return 0 <= x < self.nb_actions
