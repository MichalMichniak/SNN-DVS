import torch
import torch.nn as nn


class AddSNNLayer(nn.Module):
    def __init__(self):
        super(AddSNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0):
        output_val = input1_val + input2_val
        max_V = max(output_val)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val

    def forward(self, tj1, tj2):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        
        ti = tj1 + tj2 - 2*self.t_min  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti


class AddSNNLayer_all(nn.Module):
    """
    """
    def __init__(self, bias=0):
        super(AddSNNLayer_all, self).__init__()
        self.noise = 0
        self.B_n = 1
        self.B = bias # bias for all inputs (for Hard tanh)
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0):
        if input2_val.shape[0] != input1_val.shape[0]:
            input2_val = torch.concat((input2_val, torch.zeros(input2_val.shape)))
        output_val = input1_val + input2_val + self.B
        max_V = max(output_val)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val

    def forward(self, tj1, tj2):

        self.channels = tj1.shape[1]//2

        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        
        ti = torch.concat((tj1[0, :self.channels] + tj2[0, :self.channels] - tj1[0, self.channels:] - tj2[0, self.channels:], 
                           tj1[0, self.channels:] + tj2[0, self.channels:] - tj1[0, :self.channels] - tj2[0, :self.channels])) + self.B*(1) + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti

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

class AddSNNLayer_Htanh(nn.Module):
    def __init__(self):
        super(AddSNNLayer_Htanh, self).__init__()
        self.noise = 0
        self.B_n = 1
        self.first = AddSNNLayer_all()
        self.second = AddSNNLayer_all(1)
        self.sub = SubSNNLayer()
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0):
        tmin1, tmax1, first_val = self.first.set_params(t_min_prev, t_min,input1_val, input2_val, minimal_t_max=t_min+1)
        tmin2, tmax2, second_val = self.second.set_params(t_min_prev, t_min,input1_val, input2_val, minimal_t_max=tmax1)

        tmin1, tmax1, first_val = self.first.set_params(t_min_prev, t_min,input1_val, input2_val, minimal_t_max=tmax2)

        tmins, tmaxs, sub_val = self.sub.set_params(t_min, tmax1, first_val,second_val) ## t_min as angument do nothing
        self.sub.t_max = tmins+1
        return tmins, tmins+1, torch.minimum(sub_val,torch.ones(sub_val.shape))

    def forward(self, tj1, tj2):
        tj_first = self.first(tj1, tj2)
        tj_second = self.second(tj1, tj2)
        tj_sub = self.sub(tj_first, tj_second)
        return tj_sub