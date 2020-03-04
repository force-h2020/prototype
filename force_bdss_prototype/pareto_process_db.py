# Transferred
import numpy as np


class Pareto_process_db:

    def __init__(self, data):
        self.data = data

    def write_csv(self, fil=None):
        self.pareto_filter()
        if fil:
            np.savez(fil, self.data)
        else:
            np.savez("pareto_data.npz", volume_A_tilde=self.data[:, 0], conc_e=self.data[:, 1], temperature=self.data[:, 2], reaction_time=self.data[:, 3], impurity_conc=self.data[:, 4], prod_cost=self.data[:, 5], mat_cost=self.data[:, 6])

    def pareto_filter(self):
        idx = []
        for i in range(self.data.shape[0]):
            loc_min = False
            for j in range(self.data.shape[0]):
                lm = True
                if self.data[i, 4] <= self.data[j, 4]:
                    lm = False
                if self.data[i, 5] <= self.data[j, 5]:
                    lm = False
                if self.data[i, 6] <= self.data[j, 6]:
                    lm = False
                if lm:
                    loc_min = True
                    break
            if not loc_min:
                idx.append(i)
        idx = np.array(idx)
        self.data = self.data[idx]
        return
