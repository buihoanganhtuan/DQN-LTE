from rl2.util import np
import math

class Devices(object):
    def __init__(self, *args, **kwargs):
        """Hyper parameters of the devices and networks"""
        self.n              = int(kwargs.get('n', 500))
        self.R              = kwargs.get('R', 54)
        self.W_RAR          = kwargs.get('W_RAR', 5)
        self.N_RAR          = kwargs.get('N_RAR', math.ceil(self.R/self.W_RAR))
        self.T_RAO          = kwargs.get('T_RAO', 5)
        self.Npt_max        = kwargs.get('Npt_max', 10)
        self.m3_harq_p      = kwargs.get('m3_harq_p', 0.1)
        self.m4_harq_p      = kwargs.get('m4_harq_p', 0.1)
        self.m3_harq_max    = kwargs.get('m3_harq_max', 5)
        self.m4_harq_max    = kwargs.get('m4_harq_max', 5)
        self.T              = kwargs.get('T', 50e-3)
        self.traffic_mode   = kwargs.get('traffic_mode', 'beta')
        self.delay_thresh   = kwargs.get('delay_thresh', None)
        self.energy_thresh  = kwargs.get('energy_thresh', None)
        self.P_tx           = kwargs.get('P_tx', 50e-3)
        self.P_rx           = kwargs.get('P_rx', 50e-3)
        self.P_idle         = kwargs.get('P_idle', 0.025e-3)
        self.start_offset   = kwargs.get('start_offset', 0)
        self.end_trail      = kwargs.get('end_trail', 0)
        self.trail_counter  = self.end_trail       
        self.sf_len         = 1e-3
        self.t              = np.arange(0, int(self.T/self.T_RAO/self.sf_len))
        self.preamble_list  = np.arange(1, self.R+1)
        self.beta_p         = 60*self.t**2*(self.T/self.T_RAO/self.sf_len - self.t)**3/(self.T/self.T_RAO/self.sf_len)**6
        self.time_list      = []

        """Below are properties of the devices"""
        self.id                 = np.arange(1, self.n+1)
        self.next_msg           = np.ones(self.n, dtype=int)
        self.finish             = np.zeros(self.n, dtype=bool)
        self.blocked            = np.zeros(self.n, dtype=bool)
        self.preamble           = np.zeros(self.n, dtype=int)
        self.msg1_counter       = np.ones(self.n, dtype=int)
        self.msg3_counter       = np.zeros(self.n, dtype=int)
        self.msg4_counter       = np.zeros(self.n, dtype=int)
        self.timestamp          = np.zeros((self.Npt_max, self.n), dtype=int)
        self.tx_sfs             = np.zeros(self.n, dtype=int)
        self.rx_sfs             = np.zeros(self.n, dtype=int)
        self.delay_upto_msg3    = np.zeros(self.n, dtype=int)
        self.energy_upto_msg3   = np.zeros(self.n, dtype='float32')
        # https://stackoverflow.com/questions/46539431/np-random-choice-probabilities-do-not-sum-to-1
        if self.traffic_mode == 'beta':
            self.act_time           = np.random.choice(self.t*self.T_RAO, self.n, p=self.beta_p/np.sum(self.beta_p)) + self.start_offset * self.T_RAO
        else:
            self.act_time           = np.random.choice(self.t*self.T_RAO, self.n) + self.start_offset * self.T_RAO
        self.timer              = np.copy(self.act_time)
        self.fin_time           = np.zeros(self.n, dtype=int)
        assert ((self.delay_thresh is None) & (self.energy_thresh is None)) | ((self.delay_thresh is not None) & (self.energy_thresh is not None))
        self.rec_keys           = ['sg', 'colpre', 'tx_sfs', 'nx_sfs', 'dels']
        self.metric_keys        = ['ene', 'dels', 'block'] if self.delay_thresh is None else ['ene_meet', 'delay_meet', 'block']

    def getNearestRAO(self, t):
        if type(t) is np.ndarray:
            #assert min(temp) >= 0, 'Overflow due to extremely long barring/backoff time'
            return (np.ceil(t/self.T_RAO).astype('int32')*self.T_RAO).tolist()
        else:
            return math.ceil(t/self.T_RAO)*self.T_RAO

    def updateBlockStatus(self):
        self.blocked[self.msg1_counter > self.Npt_max] = True

    def sendPreamble(self, current_sf, control_param=None, resched_col_param=None, resched_fail_param=None):
        info = None
        rec = None
        done = False
        self.time_list = []
        # Remember: In python, logical operators take precedence over comparison operators
        # so if you write ~self.blocked & self.next_msg==1 i.e., without () wrapping the
        # self.next_msg==1, Python will intepret it as (~self.blocked & self.next_msg)==1        
        rdy = ~self.finish & ~self.blocked & (self.next_msg==1) & (self.timer==current_sf)
        arr = (self.act_time == current_sf)
        tx = self.performAccessControl(current_sf, rdy, control_param)
        if np.any(tx):
            self.tx_sfs[tx] += 1
            self.preamble[~tx] = 0
            self.preamble[tx] = np.random.choice(self.preamble_list, np.count_nonzero(tx))
            self.timestamp[self.msg1_counter[tx]-1, tx] = current_sf
            hist, _ = np.histogram(self.preamble[tx], bins=np.arange(1, self.R+2))
            sg = tx & np.isin(self.preamble, self.preamble_list[hist==1])
            d = np.zeros(sg.shape, dtype=bool)
            if np.any(sg):
                detection_prob = np.ones(self.n)
                index = np.where(sg)[0]
                temp = np.random.rand(index.shape[0])
                if np.any(temp < detection_prob[index]):
                    d[index[temp < detection_prob[index]]] = True    
                f = sg & ~d
                if np.any(f):
                    self.rx_sfs[f] += self.W_RAR
                    self.msg1_counter[f] += 1
                    self.updateBlockStatus()
                    if np.any(~self.blocked[f]):
                        f = f & ~self.blocked
                        self.rescheduleFail(current_sf, f, resched_fail_param)

            g = np.zeros(d.shape, dtype=bool)
            if np.any(d):
                index = np.random.permutation(np.where(d)[0])
                index = index[0:min(self.N_RAR*self.W_RAR, index.shape[0])]
                for ii in range(math.ceil(index.shape[0]/self.N_RAR)):
                    current_idxs = index[ii*self.N_RAR:(ii+1)*self.N_RAR]
                    self.rx_sfs[current_idxs] += ii + 1
                    self.next_msg[current_idxs] = 3
                    self.timer[current_idxs] = current_sf + 9 + ii
                    self.delay_upto_msg3[current_idxs] = self.getDelay(current_sf + 2 + ii + 1, current_idxs)
                    self.energy_upto_msg3[current_idxs] = self.getEnergy(current_sf + 2 + ii + 1, current_idxs)
                    self.time_list.append(current_sf + 9 + ii)
                g[index] = True
                #self.delay_upto_msg3[g] = self.getDelay(current_sf, g)
                #self.energy_upto_msg3[g] = self.getEnergy(current_sf, g)
                f = d & ~g
                if np.any(f):
                    self.rx_sfs[f] += self.W_RAR
                    self.msg1_counter[f] += 1
                    self.updateBlockStatus()
                    if np.any(~self.blocked[f]):
                        f = f & ~self.blocked
                        self.rescheduleFail(current_sf, f, resched_fail_param)

            c = tx & np.isin(self.preamble, self.preamble_list[hist > 1])
            if np.any(c):
                self.rx_sfs[c] += self.W_RAR
                self.msg1_counter[c] += 1
                self.updateBlockStatus()
                if np.any(~self.blocked[c]):
                    c = c & ~self.blocked
                    self.rescheduleCol(current_sf, c, resched_col_param)
            info = self.getDebugInfo(arr, rdy, tx, sg, d, g, c)
            if np.any(g):
                rec = dict(zip(self.rec_keys, [ np.count_nonzero(sg), np.count_nonzero(hist > 1), 
                                                self.tx_sfs[g], self.rx_sfs[g], self.getDelay(current_sf, g) ]))
            else:
                rec = dict(zip(self.rec_keys, [ np.count_nonzero(sg), np.count_nonzero(hist > 1), None, None, None ]))                 
        else:
            info = self.getDebugInfo(arr, rdy, tx, None, None, None, None)
            rec = dict(zip(self.rec_keys, [0, 0, None, None, None]))
        #if np.count_nonzero(self.blocked | self.finish) == self.n:
        #    self.time_list.append(-1)
        if np.count_nonzero(self.blocked | self.finish | (~self.finish & ~self.blocked & (self.next_msg != 1))) == self.n:
            if self.trail_counter == 0:
                done = True
                index = self.finish | (~self.finish & ~self.blocked & (self.next_msg != 1))
                if self.delay_thresh is None:
                    info['end_info'] = dict(zip(self.metric_keys, [self.energy_upto_msg3[index] * 1000, 
                                                                    self.delay_upto_msg3[index], np.count_nonzero(self.blocked)/self.n]))
                else:
                    info['end_info'] = dict(zip(self.metric_keys, [np.count_nonzero(self.energy_upto_msg3[index] <= self.energy_thresh)/self.n, 
                                                                    np.count_nonzero(self.delay_upto_msg3[index] <= self.delay_thresh)/self.n, np.count_nonzero(self.blocked)/self.n]))
            else:
                self.time_list.append(current_sf + self.T_RAO)
                self.trail_counter -= 1
        else:
            self.time_list.append(current_sf + self.T_RAO)
        return rec, done, info, self.time_list

    def sendMsg3(self, current_sf):
        self.time_list = []        
        tx = ~self.finish & ~self.blocked & (self.next_msg==3) & (self.timer==current_sf)
        if np.any(tx):
            s = np.copy(tx)
            s[tx] = np.random.rand(np.count_nonzero(tx)) > self.m3_harq_p
            self.msg3_counter[tx & ~s] += 1
            self.tx_sfs[tx] += 1
            self.rx_sfs[tx & ~s] += 1

            f = tx & ~s & (self.msg3_counter >= self.m3_harq_max)
            if np.any(f):
                #self.msg3_counter[f] = 0
                self.blocked[f] = True
                #self.next_msg[f] = 1
                #self.msg1_counter[f] += 1
                #self.updateBlockStatus()
                #if np.any(~self.blocked[f]):
                #    self.timer[f & ~self.blocked] = self.getNearestRAO(current_sf + 8)
            if np.any(s):
                self.msg3_counter[s] = 0
                self.next_msg[s] = 4
                self.timer[s] = current_sf + 4
                self.time_list.append(current_sf + 4)
            if np.any(tx & ~s & ~f):
                self.timer[tx & ~s & ~f] = current_sf + 8
                self.time_list.append(current_sf + 8)
        if (np.count_nonzero(self.blocked | self.finish) == self.n) and (self.trail_counter == 0):
            self.time_list.append(-1)
        return self.time_list

    def receiveMsg4(self, current_sf):
        self.time_list = []        
        tx = ~self.finish & ~self.blocked & (self.next_msg==4) & (self.timer==current_sf)
        if np.any(tx):
            s = np.copy(tx)
            s[tx] = np.random.rand(np.count_nonzero(tx)) > self.m4_harq_p
            self.msg4_counter[tx & ~s] += 1
            self.rx_sfs[tx] += 1
            self.tx_sfs[tx & ~s] += 1

            f = tx & ~s & (self.msg4_counter >= self.m4_harq_max)
            if np.any(f):
                #self.msg4_counter[f] = 0
                self.blocked[f] = True
                #self.next_msg[f] = 1
                #self.msg1_counter[f] += 1
                #self.updateBlockStatus()
                #if np.any(~self.blocked[f]):
                #    self.timer[f & ~self.blocked] = self.getNearestRAO(current_sf + 5)
            if np.any(s):
                self.finish[s] = True
                self.fin_time[s] = current_sf
            if np.any(tx & ~s & ~f):
                self.timer[tx & ~s & ~f] = current_sf + 5
                self.time_list.append(current_sf + 5)
        if (np.count_nonzero(self.blocked | self.finish) == self.n) and (self.trail_counter == 0):
            self.time_list.append(-1)      
        return self.time_list

    def refresh(self, mode, current_sf):
        assert (mode == "cont") | (mode == "episodic"), "mode should be either \"cont\" or \"episodic\""       
        index = self.finish | self.blocked
        rec = dict(zip(self.rec_keys, [0, 0, None, None, None]))
        if np.count_nonzero(index) > 0:
            # Debug message
            blocked = np.count_nonzero(self.blocked) / self.n
            mean_delay = np.mean(self.delay_upto_msg3[self.finish])
            mean_energy = np.mean(self.energy_upto_msg3[self.finish]) * 1000 # unit: mJ
            if (self.delay_thresh is not None) & (self.energy_thresh is not None):
                delay_satisfied = np.count_nonzero(self.delay_upto_msg3[self.finish] < self.delay_thresh) / self.n
                energy_satisfied = np.count_nonzero(self.energy_upto_msg3[self.finish] < self.energy_thresh) / self.n
                print('blocked: {:.4f}, mean delay: {:.4f}, mean energy: {:.4f}, delay satisfied ratio: {:.4f}, energy satisfied ratio: {:.4f}'.format( \
                        blocked, mean_delay, mean_energy, delay_satisfied, energy_satisfied))
            elif self.delay_thresh is not None:
                delay_satisfied = np.count_nonzero(self.delay_upto_msg3[self.finish] < self.delay_thresh) / self.n
                print('blocked: {:.4f}, mean delay: {:.4f}, mean energy: {:.4f}, delay satisfied ratio: {:.4f}'.format(blocked, mean_delay, \
                        mean_energy, delay_satisfied))
            elif self.energy_thresh is not None:
                energy_satisfied = np.count_nonzero(self.energy_upto_msg3[self.finish] < self.energy_thresh) / self.n
                print('blocked: {:.4f}, mean delay: {:.4f}, mean energy: {:.4f}, energy satisfied ratio: {:.4f}'.format(blocked, mean_delay, \
                        mean_energy, energy_satisfied))
            # Reset environment
            self.next_msg[index]                = 1
            if self.traffic_mode == 'beta':
                temp                            = np.random.choice(self.t*self.T_RAO, np.count_nonzero(index), p=self.beta_p/np.sum(self.beta_p)) + self.start_offset * self.T_RAO
            else:
                temp                            = np.random.choice(self.t*self.T_RAO, np.count_nonzero(index)) + self.start_offset * self.T_RAO
            if mode == "cont":
                self.act_time[index]            = self.getNearestRAO((current_sf//self.T + 1)*self.T + temp)
            elif mode == "episodic":
                self.act_time[index]            = self.getNearestRAO(temp)
            self.timer[index]                   = np.copy(self.act_time[index])
            self.finish[index]                  = False
            self.blocked[index]                 = False
            self.msg1_counter[index]            = 0
            self.msg3_counter[index]            = 0
            self.msg4_counter[index]            = 0
            self.timestamp[:, index]            = 0
            self.tx_sfs[index]                  = 0
            self.rx_sfs[index]                  = 0
            self.fin_time[index]                = 0
            self.time_list                      = self.timer[index].tolist() + np.linspace(0, self.start_offset * self.T_RAO, self.start_offset, endpoint=False).tolist()
            self.trail_counter                  = self.end_trail 
        else:
            self.time_list = [] + np.linspace(0, self.start_offset * self.T_RAO, self.start_offset, endpoint=False).tolist()
        return rec, index

    def getDelay(self, current_sf, index):
        #return self.fin_time[self.finish] - self.act_time[self.finish]
        return current_sf - self.act_time[index] + 2

    def getEnergy(self, current_sf, index):
        #return self.fin_time[self.finish] - self.act_time[self.finish]
        return ((current_sf - self.act_time[index] + 2 - self.tx_sfs[index] - self.rx_sfs[index]) * self.P_idle + \
                    self.tx_sfs[index] * self.P_tx + self.rx_sfs[index] * self.P_rx) * self.sf_len

    def getBlock(self):
        return np.count_nonzero(self.blocked)

    def getRetrans(self):
        return self.msg1_counter[self.finish]

    def addToTimelist(self, timeToAdd):
        # https://stackoverflow.com/questions/7030831/how-do-i-get-the-opposite-negation-of-a-boolean-in-python
        if not np.isscalar(timeToAdd): 
            self.time_list += timeToAdd
        else:
            self.time_list.append(timeToAdd)  

    def performAccessControl(self, current_sf, rdy, control_param):
        """ Implementing the access control method.
            If no version is implemented, then simply return the
            ready vector
        """
        return rdy

    def rescheduleCol(self, current_sf, col, resched_col_param):
        """ Implementing the rescheduling method for colliding devices. 
            This is mandatory so the base class will raise an error flag if
            this method is not implemented in a concrete implementation
        """
        raise NotImplementedError()

    def rescheduleFail(self, current_sf, fail, resched_fail_param):
        """ Implementing the rescheduling method for failed non-colliding devices. 
            This is mandatory so the base class will raise an error flag if
            this method is not implemented in a concrete implementation
        """
        raise NotImplementedError()

    def getDebugInfo(self, arr, rdy, tx, sg, d, g, c, control_param, resched_col_param, resched_fail_param):
        pass              