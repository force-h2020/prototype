# Transferred
import numpy as np

class Pareto_process_db:

    def __init__(self, data):
        self.data = data

    def write_csv(self, fil=None):
        self.pareto_filter()
        if fil:
            np.savetxt(fil, self.data, delimiter=",", fmt="%.10f")
        else:
            np.savetxt("pareto_data.csv", self.data, delimiter=",", fmt="%.12f")

    def pareto_filter(self):
        idx = []
        for i in range(self.data.shape[0]):
            loc_min = False
            for j in range(i + 1, self.data.shape[0], 1):
                lm = True
                if self.data[i, 4] < self.data[j, 4]:
                    lm = False
                if self.data[i, 5] < self.data[j, 5]:
                    lm = False
                if self.data[i, 6] < self.data[j, 6]:
                    lm = False
                if lm:
                    loc_min = True
                    break
            if not loc_min:
                idx.append(i)
        idx = np.array(idx)
        self.data = self.data[idx]
        return
