import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxMinPool2D(nn.Module):
    """
    Max Pooling or Min Pooling operation, depending on the sign of the batch normalization layer before.
    """
    def __init__(self, kernel_size, max_time, stride=None, padding=0, dilation=1):
        super(MaxMinPool2D, self).__init__()
        
        self.sign = nn.Parameter(-1*torch.ones(1, 1, 1, 1), requires_grad=False)
        self.dilation = dilation

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.max_time = max_time

    def forward(self, inputs):
        # Applying the sign to the inputs (if sign is -1, it will act as Min Pooling)
        padding_size = self.padding
        inputs = torch.nn.functional.pad(inputs, (padding_size, padding_size, padding_size, padding_size), value=self.max_time)
        pooled = F.max_pool2d(self.sign * inputs, kernel_size=self.kernel_size, stride=self.stride, padding=0, dilation=self.dilation)
        
        # Multiply the pooled result by the sign, which controls the pooling type
        return pooled * self.sign