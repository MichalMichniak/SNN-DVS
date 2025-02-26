import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from SpikingConv2D import SpikingConv2D, fuse_conv_and_bn
from Identity import IdentitySNNLayer
from Add import AddSNNLayer
from MaxMinpool2D import MaxMinPool2D
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=False))  # Changed inplace to False
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=False))  # Changed inplace to False
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)  # Changed inplace to False
        self.out_channels = out_channels

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out = out + residual
        out = F.relu(out, inplace=False)   # Use non-in-place ReLU
        return out


class ResidualSNNBlock(nn.Module):
    def __init__(self,resblock : ResidualBlock, in_channels, out_channels, stride=1, downsample=None, robustness_params = None, device = "cuda:0"):
        super(ResidualSNNBlock, self).__init__()
        conv = resblock.conv1[0]
        bn= resblock.conv1[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn)
        self.conv1 = SpikingConv2D(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)
        self.device = device
        
        conv = resblock.conv2[0]
        bn= resblock.conv2[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn)
        self.conv2 = SpikingConv2D(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)

        self.downsample = downsample
        self.identity = IdentitySNNLayer()
        self.add_layer = AddSNNLayer()
        self.out_channels = out_channels

    def set_params(self, t_min_prev, t_min, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        t_min1, t_max1 = self.conv1.set_params(t_min_prev=t_min_prev,t_min=t_min)
        self.pooling1 = MaxMinPool2D(2, t_max1.data,2).to(self.device)
        t_min2, t_max2 = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1)
        max_out2 = t_max2 - t_min2
        
        if self.downsample:
            t_min_dummy, t_max1_dummy = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2)
            self.pooling2 = MaxMinPool2D(2, t_max1_dummy.data,2).to(self.device)
        else:
            t_min_dummy, t_max1_dummy = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2)

        t_min2, t_max2 = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, minimal_t_max=t_max1_dummy)
        
        # time t_max2 and t_max1_dummy are the same
        t_min_add = t_max2 - max(max_dummy1, max_out2)

        self.t_min, self.t_max = self.add_layer.set_params(t_min_add, t_max2)
        return self.t_min, self.t_max


    def forward(self, x):
        residual = x
        out = self.conv1(x)
        if self.downsample:
            residual = self.downsample(x)
            residual = self.pooling2(residual)
            out = self.pooling1(out)
        else:
            residual = self.identity(x)
        out = self.conv2(out)
        out = self.add_layer(out,residual)
        return out