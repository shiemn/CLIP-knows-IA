from typing import Sequence, Union

import torch
import torch.nn as nn

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce

__all__ = ["EarthMoverDistance"]

class EarthMoverDistance(Metric):
    
    def __init__(self,  device: Union[str, torch.device] = torch.device("cpu")):
        super(EarthMoverDistance).__init__()
        self._device = device
        self._num_examples = 0
        self._sum_of_distances = torch.tensor(0.0, device=self._device)

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_distances = torch.tensor(0.0, device=self._device)
        self._num_examples = 0


    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y = output[0].detach(), output[1].detach()
        self._sum_of_distances += torch.square(torch.cumsum(y, dim=-1) - torch.cumsum(y_pred, dim=-1)).to(self._device)
        self._num_examples += y.shape[0]


    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("EarthMoverDistance must have at least one example before it can be computed.")
        return self._sum_of_distances.item() / self._num_examples



""" def earth_mover_distance(y_true, y_pred):
    return torch.mean(torch.square(torch.cumsum(y_true, dim=-1) - torch.cumsum(y_pred, dim=-1)), dim=-1) """


class EMD_Loss(nn.Module):


    def __init__(self) -> None:
        super(EMD_Loss, self).__init__()


    def single_emd_loss(self, p, q, r=2):
        """
        Earth Mover's Distance of one sample
        Args:
            p: true distribution of shape num_classes × 1
            q: estimated distribution of shape num_classes × 1
            r: norm parameter
        """
        assert p.shape == q.shape, "Length of the two distribution must be the same"
        length = p.shape[0]
        emd_loss = 0.0
        for i in range(1, length + 1):
            emd_loss += torch.abs(sum(p[:i] - q[:i])) ** r
        return (emd_loss / length) ** (1. / r)


    def emd_loss(self, p, q, r=2):
        """
        Earth Mover's Distance on a batch
        Args:
            p: true distribution of shape mini_batch_size × num_classes × 1
            q: estimated distribution of shape mini_batch_size × num_classes × 1
            r: norm parameters
        """
        assert p.shape == q.shape, "Shape of the two distribution batches must be the same."
        mini_batch_size = p.shape[0]
        loss_vector = []
        for i in range(mini_batch_size):
            loss_vector.append(self.single_emd_loss(p[i], q[i], r=r))
        return sum(loss_vector) / mini_batch_size


    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.emd_loss(target, input)
