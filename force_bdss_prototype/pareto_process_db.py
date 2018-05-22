# Transferred
import numpy as np

class Pareto_process_db:

    def __init__(self, data):
        self.data = data

    def write_csv(self, fil=None):
        if fil:
            np.savetxt(fil, self.data, delimiter=";", fmt="%.10f")
        else:
            np.savetxt("pareto_data.csv", self.data, delimiter=",", fmt="%.10f")
