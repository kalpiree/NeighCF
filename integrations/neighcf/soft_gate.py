import math
import torch
from torch import nn


class SoftNeighborEmbeddingGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.policy = config.threshold_policy
        self.temperature = config.gate_temperature
        self.user_positions = {user: index for index, user in enumerate(sorted(config.user_thresholds))}
        if self.policy == 'activity_logistic':
            initial = [config.activity_threshold_intercept, config.activity_threshold_slope]
        elif self.policy == 'per_user':
            initial = [self._logit(config.user_thresholds[user]) for user in sorted(config.user_thresholds)]
        else:
            initial = [self._logit(config.threshold_value)]
        self.parameters_raw = nn.Parameter(torch.tensor(initial, dtype=torch.float32))

    @staticmethod
    def _logit(value):
        if not 0 < value < 1:
            raise ValueError('learned threshold initialization must be strictly between 0 and 1')
        return math.log(value / (1 - value))

    def threshold(self, user_id, history_length):
        if self.policy == 'activity_logistic':
            raw = self.parameters_raw[0] + self.parameters_raw[1] * math.log1p(history_length)
        elif self.policy == 'per_user':
            if user_id not in self.user_positions:
                raise ValueError('learned per-user threshold missing for query user')
            raw = self.parameters_raw[self.user_positions[user_id]]
        else:
            raw = self.parameters_raw[0]
        return torch.sigmoid(raw)

    def forward(self, scores, user_id, history_length):
        values = torch.as_tensor(scores, dtype=self.parameters_raw.dtype, device=self.parameters_raw.device)
        return torch.sigmoid((values - self.threshold(user_id, history_length)) / self.temperature)
