

import torch.nn as nn
from SNN.SpikingConv2D import fuse_conv_and_bn
import torch.nn.functional as F
from SNN.SpikingConv2D_Htanh import SpikingConv2D_Htanh, SpikingConv2D_all
from SNN.Identity_INN import IdentitySNNLayer
from SNN.MaxMinpool2D import MaxMinPool2D
from SNN.Add_INN import AddSNNLayer_all, AddSNNLayer_Htanh
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, end_maxpool = False):
        super(ResidualBlock, self).__init__()
        if(downsample is not None):
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(inplace=False),
                            nn.MaxPool2d(kernel_size=2, stride=2)
                            )  # Changed inplace to False
        else:
            self.conv1 = nn.Sequential(
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding='same'),
                            nn.BatchNorm2d(out_channels),
                            nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)
                            )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False))  # Changed inplace to False
        self.downsample = downsample
        self.relu = nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=False)  # Changed inplace to False
        self.out_channels = out_channels
        self.end_maxpool = end_maxpool

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        if self.end_maxpool:
            out = F.relu(out, inplace=False)
        else:
            out = F.hardtanh(out, inplace=False, min_val=-1.0, max_val=1.0)   # Use non-in-place ReLU
        return out


class ResidualSNNBlock_all(nn.Module):
    def __init__(self,resblock : ResidualBlock, in_channels, out_channels, stride=1, downsample=None, robustness_params = None, device = "cuda:0", end_maxpool = False):
        super(ResidualSNNBlock_all, self).__init__()
        conv = resblock.conv1[0]
        bn= resblock.conv1[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn, device=device)
        if (downsample is not None):
            self.conv1 = SpikingConv2D_all(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)
        else:
            self.conv1 = SpikingConv2D_Htanh(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)
        self.device = device
        
        conv = resblock.conv2[0]
        bn= resblock.conv2[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn, device=device)
        self.conv2 = SpikingConv2D_Htanh(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)

        self.downsample = downsample
        self.identity = IdentitySNNLayer()
        if end_maxpool:
            self.add_layer = AddSNNLayer_all()
        else:
            self.add_layer = AddSNNLayer_Htanh()
        self.out_channels = out_channels
        self.end_maxpool = end_maxpool

    def set_params(self, t_min_prev, t_min, input_val, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        t_min1, t_max1, conv1_val = self.conv1.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val)
        self.t_max1 = t_max1
        self.pooling1 = MaxMinPool2D(2, t_max1.data,2).to(self.device)
        t_min2, t_max2, conv2_val = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, in_ranges_max=conv1_val)
        max_out2 = t_max2 - t_min2
        
        if self.downsample:
            t_min_dummy, t_max1_dummy, downsample_val = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val)
            self.t_max1_dummy = t_max1_dummy
            self.pooling2 = MaxMinPool2D(2, t_max1_dummy.data,2).to(self.device)
        else:
            t_min_dummy, t_max1_dummy, downsample_val = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val)

        t_min2, t_max2, conv2_val = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, minimal_t_max=t_max1_dummy, in_ranges_max=conv1_val)
        
        # time t_max2 and t_max1_dummy are the same
        t_min_add = t_max2 - max(max_dummy1, max_out2)

        self.t_min, self.t_max, add_val = self.add_layer.set_params(t_min_add, t_max2, conv2_val, downsample_val)

        self.times = [(t_min1, t_max1, 'c'), (t_min2, t_max2, 'c'), (self.t_min, self.t_max, 'a') ]
        return self.t_min, self.t_max, add_val

    def get_main_times(self):
        return self.times

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        out = self.conv1(x)
        if self.downsample:
            residual = self.downsample(x)
            residual = self.pooling2(residual)
            out = self.pooling1(out)
            residual = torch.concat((residual,torch.ones(residual.shape)*self.t_max1_dummy), dim=1)
            out = torch.concat((out,torch.ones(out.shape)*self.t_max1), dim=1)
        else:
            residual = self.identity(x)
        # print(out.shape)
        out = self.conv2(out)
        out = self.add_layer(out,residual) # no need for adding negative part
        return out


class LayerSNN_all(nn.Module):
    def __init__(self, layer, inplanes, planes, blocks, stride=1, device = 'cuda:0', end_maxpool = False):
        super(LayerSNN_all, self).__init__()
        self.inplanes = inplanes

        downsample = None
        if stride != 1 or self.inplanes != planes:
            conv2d, bias_from_nn = layer[0].downsample[0], layer[0].downsample[1]
            conv_fused = fuse_conv_and_bn(conv2d, bias_from_nn, device=device)
            robustness_params={
                'noise':0.0,
                'time_bits':0,
                'weight_bits': 0,
                'latency_quantiles':0.0
            }
            downsample = SpikingConv2D_all(planes,"test2", padding='same', stride=1, device=device,robustness_params=robustness_params,kernels=conv_fused.weight.data, biases=conv_fused.bias, kernel_size=(1,1))
            # t_min, t_max = spiking_conv2.set_params(0,1)stride
            
        self.layers = []
        self.layers.append(ResidualSNNBlock_all(layer[0],self.inplanes,planes, 1, downsample=downsample, device=device))
        self.inplanes = planes
        self.blocks = blocks
        for i in range(1, blocks):
            if i == blocks-1 and end_maxpool:
                self.layers.append(ResidualSNNBlock_all(layer[i],self.inplanes,planes, 1, downsample=None, device=device, end_maxpool = True))
            else:
                self.layers.append(ResidualSNNBlock_all(layer[i],self.inplanes,planes, 1, downsample=None, device=device))
            
        
    
    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0):
        tmin, tmax = t_min_prev, t_min
        for i in range(self.blocks):
            tmin, tmax, in_ranges_max = self.layers[i].set_params(tmin, tmax,in_ranges_max)
        return tmin, tmax, in_ranges_max
    
    def get_main_times(self):
        lst = []
        for i in range(self.blocks):
            lst.extend(self.layers[i].get_main_times())
        return lst
    
    def forward(self, x):
        for i in range(self.blocks):
            # print(i)
            x = self.layers[i].forward(x)
        return x