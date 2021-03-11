from rl2.devices import Devices

import numpy as np

class ACB_Devices(Devices):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.acb_fail_counter   = np.zeros(self.n, dtype=int)
        self.control_dim = None

        self.SI_win_len = kwargs.get('SI_win_len', 2)

        if not kwargs['control_backoff']:
            self.backoff_mode = kwargs.get('bo_mode', True) # True = normal backoff, False = exponential backoff
            if self.backoff_mode:
                self.backoff_ind = kwargs.get('bi', 0)
        
        if not kwargs['control_tbar']:
            self.acb_mode = kwargs.get('acb_mode', True) # True = normal ACB, False = exponential ACB
            if self.acb_mode:
                self.T_bar = kwargs.get('T_bar', 0)
        
        self.backlog_log = kwargs.get('backlog_log', False)       
        if self.backlog_log:
            self.info_keys = ['nb_arr', 'nb_tx', 'nb_suc', 'nb_backlog']
        else:
            self.info_keys = ['nb_arr', 'nb_tx', 'nb_suc']    

    def performAccessControl(self, current_sf, rdy, control_param):
        if self.control_dim is None:
            self.control_dim = len(control_param)
        self.rx_sfs[rdy] += self.SI_win_len
        tx = np.copy(rdy)
        tx[rdy] = np.random.rand(np.count_nonzero(rdy)) < control_param if self.control_dim==1 else np.random.rand(np.count_nonzero(rdy)) < control_param[0]
        if np.any(rdy & ~tx):
            f = rdy & ~tx
            self.acb_fail_counter[f] += 1
            if self.control_dim == 1:
                if self.acb_mode:
                    temp = (0.7 + 0.6 * np.random.random_sample(np.count_nonzero(f))) * self.T_bar if self.T_bar != 0 else 1
                else:
                    temp = (0.7 + 0.6 * np.random.random_sample(np.count_nonzero(f))) * 2**self.acb_fail_counter[f]
                temp = self.getNearestRAO(current_sf + temp)
                self.timer[rdy & ~tx] = temp
                self.addToTimelist(temp)                    
            elif self.control_dim == 2:
                temp = (0.7 + 0.6 * np.random.random_sample(np.count_nonzero(f))) * control_param[1] if control_param[1] != 0 else 1
                temp = self.getNearestRAO(current_sf + temp)
                self.timer[rdy & ~tx] = temp
                self.addToTimelist(temp)              
            elif self.control_dim == 3:
                f0 = f & (self.acb_fail_counter == 0)
                f1 = f & (self.acb_fail_counter > 0)
                temp0 = (0.7 + 0.6 * np.random.random_sample(np.count_nonzero(f0))) * control_param[1] if control_param[1] != 0 else 1
                temp1 = (0.7 + 0.6 * np.random.random_sample(np.count_nonzero(f1))) * control_param[2] if control_param[2] != 0 else 1
                temp0 = self.getNearestRAO(current_sf + temp0)
                temp1 = self.getNearestRAO(current_sf + temp1)
                self.timer[f0] = temp0
                self.timer[f1] = temp1
                self.addToTimelist(temp0)
                self.addToTimelist(temp1)
            else:
                raise NotImplementedError('action space dimension not supported')

        return tx

    def rescheduleCol(self, current_sf, c, resched_col_param):
        if resched_col_param is None:
            if self.backoff_mode:
                temp = np.random.randint(0, self.backoff_ind+1, size=np.count_nonzero(c)) if self.backoff_ind != 0 else 0
            else:
                temp = np.random.randint(0, 10*2**(self.msg1_counter[c] - 1), size=np.count_nonzero(c))
        else:
            temp = np.random.randint(0, resched_col_param+1, size=np.count_nonzero(c))
        temp = self.getNearestRAO(current_sf + 2 + self.W_RAR + 1 + temp)
        self.timer[c] = temp
      
        return

    def rescheduleFail(self, current_sf, f, resched_fail_param):
        # In conventional LTE, fail devices are treated in the same way as colliding devices
        self.rescheduleCol(current_sf, f, resched_fail_param)
        #self.timer[f] = self.getNearestRAO(current_sf + 2 + self.W_RAR + 1)
        #self.time_list.append(self.getNearestRAO(current_sf + 2 + self.W_RAR + 1))
    
    def getDebugInfo(self, arr, rdy, tx, sg, d, g, c):
        if np.any(tx):
            if self.backlog_log:
                info = dict(zip(self.info_keys, [*np.count_nonzero([arr, tx, g, rdy], axis=-1).tolist()]))
            else:
                info = dict(zip(self.info_keys, [*np.count_nonzero([arr, tx, g], axis=-1).tolist()]))
        else:
            if self.backlog_log:
                info = dict(zip(self.info_keys, [np.count_nonzero(arr), 0, 0, np.count_nonzero(rdy)]))
            else:
                info = dict(zip(self.info_keys, [np.count_nonzero(arr), 0, 0]))
        return info

    def refresh(self, mode, current_sf):
        rec, index = super().refresh(mode, current_sf)
        if np.count_nonzero(index) > 0:
            self.acb_fail_counter[index] = 0
        return rec, self.time_list