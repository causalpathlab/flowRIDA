

'''
modified from-
https://github.com/AxelNathanson/pytorch-normalizing-flows/blob/main/flow_models.py

'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ConditionalAffineCoupling(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim):
        super().__init__()
        self.scale_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.shift_net = nn.Sequential(
            nn.Linear(input_dim + condition_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, u, condition):
        combined = torch.cat([u, condition], dim=-1)
        scale = self.scale_net(combined)
        scale = torch.tanh(scale)  # Stabilize the scaling factor
        shift = self.shift_net(combined)
        w = u * torch.exp(scale) + shift
        log_det = scale.sum(dim=-1)  # Log determinant of the scaling factors
        return w, log_det

    def reverse(self, w, condition):
        combined = torch.cat([w, condition], dim=-1)
        scale = self.scale_net(combined)
        shift = self.shift_net(combined)
        scale = torch.tanh(scale)
        u = (w - shift) * torch.exp(-scale)
        return u


class SimpleFlowModel(nn.Module):
    def __init__(self, input_dim, condition_dim, hidden_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([
            ConditionalAffineCoupling(input_dim, condition_dim, hidden_dim)
            for _ in range(num_layers)
        ])

    def forward(self, u, condition):
        total_log_det = 0
        for layer in self.layers:
            u, log_det = layer(u, condition)
            total_log_det += log_det
        return u, total_log_det

    def reverse(self, w, condition):
        for layer in reversed(self.layers):
            w = layer.reverse(w, condition)
        return w


