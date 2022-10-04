from ignite.metrics import Metric
from ignite.exceptions import NotComputableError

import numpy as np
from scipy.stats import pearsonr

# These decorators helps with distributed settings
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced


class PearsonCorr(Metric):

    def __init__(self, output_transform=lambda x: x, device="cpu"):
        self._num_examples = None
        self._samples_x = [] #Predictions
        self._samples_y = [] #Actual Truth Values
        super(PearsonCorr, self).__init__(output_transform=output_transform, device=device)

    @reinit__is_reduced
    def reset(self):
        self._samples_x = []
        self._samples_y = []
        self._num_examples = 0
        super(PearsonCorr, self).reset()

    @reinit__is_reduced
    def update(self, output):
        x, y = output[0].detach().tolist(), output[1].detach().tolist()

        self._samples_x.extend(x)
        self._samples_y.extend(y)


    @sync_all_reduce()
    def compute(self):
        #np.array(self._samples_x)
        #np.array(self._samples_y)
        
        r,p = pearsonr(np.array(self._samples_x), np.array(self._samples_y))
        return r
        
