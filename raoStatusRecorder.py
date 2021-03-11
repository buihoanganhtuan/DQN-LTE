class raoStatusRecorder(object):
    def __init__(self, *args, **kwargs):
        self.subframe_num       = []
        self.num_sing_pre       = []
        self.num_col_pre        = []
        self.real_num_tx        = []
        super().__init__(*args, **kwargs)

    def record(self, subframe_num, num_sing_pre, num_col_pre, real_num_tx):
        self.subframe_num.append(subframe_num)
        self.num_sing_pre.append(num_sing_pre)
        self.num_col_pre.append(num_col_pre)
        self.real_num_tx.append(real_num_tx)
        return None

