import torch.nn as nn
import torch

class SubSNNLayer(nn.Module):
    def __init__(self):
        super(SubSNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0):
        output_val = input1_val
        max_V = max(output_val)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val

    def forward(self, tj1, tj2):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        
        ti = tj1 - self.t_min - (tj2 - self.t_min)  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti