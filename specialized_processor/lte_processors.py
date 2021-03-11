import numpy as np

from rl2.core import Processor

class LteProcessor(Processor):
    def __init__(self):
        super().__init__()
        self.max_reward = 1
        self.max_nb_col_pre = 1
        self.max_nb_sg_pre = 1

    def process_observation(self, observation):
        if observation[0] > self.max_nb_sg_pre:
            self.max_nb_sg_pre = observation[0]
            self.max_reward = observation[0]
        if observation[1] > self.max_nb_col_pre:
            self.max_nb_col_pre = observation[1]
        return [observation[0]/self.max_nb_sg_pre, observation[1]/self.max_nb_col_pre, observation[-1]]

    def process_state_batch(self, batch):
        # Remember, a state is a sequence of consecutive observations
        # We already process the observation before storing, so we DON'T need 
        # to process state batch (taken from memory)
        #batch[:, :, 0] /= self.max_nb_sg_pre
        #batch[:, :, 1] /= self.max_nb_col_pre
        return batch

    def process_reward(self, reward):
        if reward[0] > self.max_reward:
            self.max_reward = reward[0]
        return reward[0] / self.max_reward

    def process_info(self, info):
        #info = {
            #'hist_reward': self.max_reward,
            #'hist_nb_col_pre': self.max_nb_col_pre,
        #    **info
        #}
        return info

class LteProcessorConstraint(Processor):
    def __init__(self, delay_thresh, energy_thresh, ratio, control_backoff, control_tbar):
        super().__init__()
        self.P_idle = 0.025e-3
        self.P_tx   = 50e-3
        self.P_rx   = 50e-3
        self.sf_len = 1e-3      
        self.max_reward = 1
        self.max_nb_col_pre = 1
        self.max_nb_sg_pre = 1
        self.max_delay_reward = 1
        self.max_energy_reward = 1
        self.max_avg_del = 1
        self.max_avg_energy = self.P_idle * self.sf_len
        self.delay_thresh = delay_thresh
        self.energy_thresh = energy_thresh
        self.ratio = ratio
        self.control_backoff = control_backoff
        self.control_tbar = control_tbar

    def process_observation(self, observation):
        # Remember that the passed in observation
        # has been already modified by the ACB environment
        # [Ns, Nc, tx_sfs, rx_sfs, dels, p_bar, bo]
        # Process this observation to convert it to the form of
        # [Ns, Nc, Pacb, bo] or [Ns, Nc, Davg, Evag, Pacb, bo] or w/o bo

        processed_observation = []
        if observation[0] > self.max_nb_sg_pre:
            self.max_nb_sg_pre = observation[0]
            self.max_reward = observation[0]
        #processed_observation.append(observation[0]/self.max_nb_sg_pre)
        processed_observation.append(observation[0])

        if observation[1] > self.max_nb_col_pre:
            self.max_nb_col_pre = observation[1]
        #processed_observation.append(observation[1]/self.max_nb_col_pre)
        processed_observation.append(observation[1])

        if observation[4] is None:
            processed_observation.append(0.)
        else:
            temp = np.mean(observation[4])
            if temp > self.max_avg_del:
                self.max_avg_del = temp
            #processed_observation.append(temp/self.max_avg_del)            
            processed_observation.append(temp)            

        if observation[2] is None:
            processed_observation.append(0.)
        else:
            temp = observation[2] * self.P_tx + observation[3] * self.P_rx + \
                    (observation[4] - observation[3] - observation[2]) * self.P_idle
            temp = np.mean(temp) * self.sf_len
            if temp > self.max_avg_energy:
                self.max_avg_energy = temp
            #processed_observation.append(temp/self.max_avg_energy)            
            processed_observation.append(temp * 1000)

        # appending the ACB factor
        if self.control_backoff | self.control_tbar:
            processed_observation.append(observation[-2])
            #processed_observation.append(observation[-1]/960)
            processed_observation.append(observation[-1])
        else:
            processed_observation.append(observation[-2])
        return processed_observation

    def process_reward(self, reward):
        # Reward input is in the form of (Ns, Nc, tx_sfs, rx_sfs, dels)
        # (raw observation)
        if reward[2] is None:
            return 0.
        else:
            nb_devices = len(reward[4])
            if nb_devices == 0:
                return 0.
            else:
                temp = (reward[2] * self.P_tx + reward[3] * self.P_rx + (reward[4] - reward[3] - reward[2]) * self.P_idle) * self.sf_len
                delay_reward = np.count_nonzero(reward[4] < self.delay_thresh)
                energy_reward = np.count_nonzero(temp < self.energy_thresh)
                if delay_reward > self.max_delay_reward:
                    self.max_delay_reward = delay_reward
                if energy_reward > self.max_energy_reward:
                    self.max_energy_reward = energy_reward
                processed_reward = self.ratio * delay_reward + (1. - self.ratio) * energy_reward
                #processed_reward = self.ratio * delay_reward / self.max_delay_reward + (1. - self.ratio) * energy_reward / self.max_energy_reward
                #processed_reward = self.ratio * delay_reward / nb_devices + (1. - self.ratio) * energy_reward / nb_devices
                return processed_reward

class LteProcessorConstraintNew(Processor):
    def __init__(self, delay_thresh, energy_thresh, ratio, action_space_size):
        super().__init__()
        self.P_idle = 0.025e-3
        self.P_tx   = 50e-3
        self.P_rx   = 50e-3
        self.sf_len = 1e-3      
        self.max_avg_energy = self.P_idle * self.sf_len
        self.delay_thresh = delay_thresh
        self.energy_thresh = energy_thresh
        self.ratio = ratio
        self.action_space_size = action_space_size

    def process_observation(self, observation):
        # Remember that the passed in observation
        # has been already modified by the ACB environment
        # Updated: the passed in (ACB env-modified) observation is now of the form [Ns, Nc, tx_sfs, rx_sfs, dels, action_index]
        # processsed it to get the form [Ns, Nc, Davg, Eavg, action_index]
        assert len(observation) == 6, 'Unsupported ACB-env-formatted observation format'

        processed_observation = []
        processed_observation.append(observation[0])
        processed_observation.append(observation[1]) 

        if observation[4] is None:
            processed_observation.append(0.)
        else:        
            processed_observation.append(np.mean(observation[4]))    

        if observation[2] is None:
            processed_observation.append(0.)
        else:
            temp = observation[2] * self.P_tx + observation[3] * self.P_rx + \
                    (observation[4] - observation[3] - observation[2]) * self.P_idle
            temp = np.mean(temp) * self.sf_len          
            processed_observation.append(temp * 1000)     

        # appending the normalized action_index
        processed_observation.append(observation[5] / self.action_space_size)
        return processed_observation

    def process_reward(self, reward):
        # Reward input is in the form of (Ns, Nc, tx_sfs, rx_sfs, dels)
        # (raw observation)
        assert len(reward) == 5, 'Unsupported raw observation format'
        if reward[2] is None:
            return 0.
        else:
            nb_devices = len(reward[4])
            if nb_devices == 0:
                return 0.
            else:
                temp = (reward[2] * self.P_tx + reward[3] * self.P_rx + (reward[4] - reward[3] - reward[2]) * self.P_idle) * self.sf_len
                delay_reward = np.count_nonzero(reward[4] < self.delay_thresh)
                energy_reward = np.count_nonzero(temp < self.energy_thresh)
                processed_reward = self.ratio * delay_reward + (1. - self.ratio) * energy_reward
                return processed_reward

class LteProcessorConstraintNew2(Processor):
    def __init__(self, delay_thresh, energy_thresh, ratio, action_space_size):
        super().__init__()
        self.P_idle = 0.025e-3
        self.P_tx   = 50e-3
        self.P_rx   = 50e-3
        self.sf_len = 1e-3      
        self.max_avg_energy = self.P_idle * self.sf_len
        self.delay_thresh = delay_thresh
        self.energy_thresh = energy_thresh
        self.ratio = ratio
        self.action_space_size = action_space_size

    def process_observation(self, observation):
        # Remember that the passed in observation
        # has been already modified by the ACB environment
        # Updated: the passed in (ACB env-modified) observation is now of the form [Ns, Nc, tx_sfs, rx_sfs, dels, action_index]
        # processsed it to get the form [Ns, Nc, Davg, Eavg, action_index]
        assert len(observation) == 6, 'Unsupported ACB-env-formatted observation format'

        processed_observation = []
        processed_observation.append(observation[0] / 54)
        processed_observation.append(observation[1] / 54) 

        if observation[4] is None:
            processed_observation.append(0.)
        else:        
            processed_observation.append(np.mean(observation[4]) / 2000)    

        if observation[2] is None:
            processed_observation.append(0.)
        else:
            temp = observation[2] * self.P_tx + observation[3] * self.P_rx + \
                    (observation[4] - observation[3] - observation[2]) * self.P_idle
            temp = np.mean(temp) * self.sf_len          
            processed_observation.append(temp * 1000)     

        # appending the normalized action_index
        processed_observation.append(observation[5] / self.action_space_size)
        return processed_observation

    def process_reward(self, reward):
        # Reward input is in the form of (Ns, Nc, tx_sfs, rx_sfs, dels)
        # (raw observation)
        assert len(reward) == 5, 'Unsupported raw observation format'
        if reward[2] is None:
            return 0.
        else:
            nb_devices = len(reward[4])
            if nb_devices == 0:
                return 0.
            else:
                temp = (reward[2] * self.P_tx + reward[3] * self.P_rx + (reward[4] - reward[3] - reward[2]) * self.P_idle) * self.sf_len
                delay_reward = np.count_nonzero(reward[4] < self.delay_thresh)
                energy_reward = np.count_nonzero(temp < self.energy_thresh)
                processed_reward = self.ratio * delay_reward + (1. - self.ratio) * energy_reward
                return processed_reward / 54