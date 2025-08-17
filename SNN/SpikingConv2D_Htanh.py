from  SNN.SpikingConv2D_INN import SpikingConv2D
from SNN.SubSNNLayer import SubSNNLayer
import torch.nn as nn
import torch

class SpikingConv2D_Htanh(nn.Module):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params=None, kernels = None, device = 'cuda:0', biases = None, stride=1):
        super(SpikingConv2D_Htanh, self).__init__()
        if robustness_params is None:
            robustness_params = {}
        kernels_pos = torch.concat((kernels,-kernels), dim=1 )
        if biases is not None:
            biases_pos = biases
        else:
            biases_pos = None
        
        kernels_neg = torch.concat((-kernels,kernels), dim=1 )
        if biases is not None:
            biases_neg = -biases
        else:
            biases_neg = None

        kernels_new = torch.concat((kernels_pos, kernels_neg), dim=0)
        biases_new = torch.concat((biases_pos, biases_neg))
        # print(biases_new.shape)
        self.conv_first = SpikingConv2D(2*filters, name, device = device, padding=padding, stride=stride, kernel_size=kernel_size,robustness_params=robustness_params, kernels=kernels_new, biases=biases_new)
        
        kernels_new2 = torch.concat((kernels_pos, kernels_neg), dim=0)
        biases_new2 = torch.concat((biases_pos, biases_neg)) - 1
        self.conv_second = SpikingConv2D(2*filters, name, device = device, padding=padding, stride=stride, kernel_size=kernel_size,robustness_params=robustness_params, kernels=kernels_new2, biases=biases_new2)
        self.sub = SubSNNLayer()
        self.filters = filters*2

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        if(in_ranges_max.shape[0] != self.filters):
            in_ranges_max = torch.concat((in_ranges_max,torch.zeros(in_ranges_max.shape)))
        tmin1, tmax1, first_val = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max, minimal_t_max-1)
        tmin2, tmax2, second_val = self.conv_second.set_params(t_min_prev, t_min, in_ranges_max, tmax1)

        tmin1, tmax1, first_val = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max, tmax2)
        self.t_max = tmax1
        tmins, tmaxs, sub_val = self.sub.set_params(t_min, tmax1, first_val,second_val)
        tmaxs = tmins+1
        self.sub.t_max = tmaxs
        self.t_max = tmaxs
        # Returning for function signature consistency
        return tmins, self.t_max, torch.minimum(sub_val, torch.ones(sub_val.shape))
        # return tmin1, self.t_max, torch.minimum(first_val, torch.ones(first_val.shape))

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        tj1 = self.conv_first(tj)
        tj2 = self.conv_second(tj)
        tj_sub = self.sub(tj1, tj2)

        return tj_sub
        # return tj1

class SpikingConv2D_all(nn.Module):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params=None, kernels = None, device = 'cuda:0', biases = None, stride=1):
        super(SpikingConv2D_all, self).__init__()
        if robustness_params is None:
            robustness_params = {}
        kernels_pos = torch.concat((kernels,-kernels), dim=1 )
        if biases is not None:
            biases_pos = biases
        else:
            biases_pos = None
        kernels_new = kernels_pos
        biases_new = biases_pos
        self.conv_first = SpikingConv2D(filters, name, device = device, padding=padding, stride=stride, kernel_size=kernel_size,robustness_params=robustness_params, kernels=kernels_new, biases=biases_new)

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        tmin1, tmax1, first_val = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max)
        
        # Returning for function signature consistency
        return tmin1, tmax1, first_val
        # return tmin1, self.t_max, torch.minimum(first_val, torch.ones(first_val.shape))

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        tj1 = self.conv_first(tj)
        return tj1
        # return tj1