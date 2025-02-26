import torch
import torch.nn as nn

class AddSNNLayer(nn.Module):
    def __init__(self):
        super(AddSNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        pass

    def set_params(self, t_min_prev, t_min, minimal_t_max = 0):
        max_input = (t_min - t_min_prev)
        max_V = max_input * 2
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max)

    def forward(self, tj1, tj2):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        
        ti = tj1 + tj2 - 2*self.t_min  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti