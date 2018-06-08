import subprocess
import sys
import itertools
import collections

from force_bdss.api import BaseMCO


def rotated_range(start, stop, starting_value):
    """Produces a range of integers, then rotates so that the starting value
    is starting_value"""
    r = list(range(start, stop))
    start_idx = r.index(starting_value)
    d = collections.deque(r)
    d.rotate(-start_idx)
    return list(d)


class MCO(BaseMCO):
    def run(self, model):
        parameters = model.parameters

        values = []
        for p in parameters:
            values.append(
                rotated_range(int(p.lower_bound),
                              int(p.upper_bound),
                              int(p.initial_value))
            )

        value_iterator = itertools.product(*values)

        self.started = True

        for value in value_iterator:
            # When there is new data, this operation informs the system that
            # new data has been received. It must be a dictionary as given.
            self.new_data = {
                'input': tuple(value),
                'output': tuple(out_data)
            }

        # To inform the rest of the system that the evaluation has completed.
        # we set this event to True
        self.finished = True

    def calculate_singlepoint(self, in_values):
        application = self.factory.plugin.application

        ps = subprocess.Popen(
            [sys.argv[0],
             "--evaluate",
             application.workflow_filepath],
            stdout=subprocess.PIPE,
            stdin=subprocess.PIPE)

        out = ps.communicate(
            " ".join([str(v) for v in in_values]).encode("utf-8"))

        return out[0].decode("utf-8").split()
