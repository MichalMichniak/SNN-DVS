import torch
from torch import nn
import torch.nn.functional as F

L = 20 # multiplier coef
eps = 0.0
eps_V = 0.00001

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

def call_spiking(tj, W, D_i, t_min, t_max, noise, dtype=torch.FloatTensor):
    """
    Calculates spiking times to recover ReLU-like functionality.
    Assumes tau_c=1 and B_i^(n)=1.
    """
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    
    #### Check ####
    V = torch.matmul((tj - t_min).type(dtype), torch.maximum(W.type(dtype), torch.zeros(W.type(dtype).shape)))
    if((V>threshold).any()):
        print(f"ERROR SpikingDense V {V}, thr {threshold}") 
    ### END Check ###

    # Calculate output spiking time ti (Eq. 7)
    ti = torch.matmul((tj - t_min).type(dtype), W.type(dtype)) + threshold + t_min
    
    # Ensure valid spiking time: do not spike for ti >= t_max
    ti = torch.where(ti < t_max, ti, t_max)

    # Add noise to the spiking time for noise simulations
    if noise > 0:
        ti = ti + torch.randn_like(ti) * noise
    
    return ti

class SpikingDense(nn.Module):
    def __init__(self, units, name, X_n=1, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        super().__init__()
        self.units = units
        self.B_n = (1 + 0.0) * X_n
        self.outputLayer=outputLayer
        self.t_min_prev, self.t_min, self.t_max=0, 0, 1
        self.noise=robustness_params['noise']
        self.time_bits=robustness_params['time_bits']
        self.weight_bits =robustness_params['weight_bits'] 
        self.w_min, self.w_max=-1.0, 1.0
        self.alpha = torch.full((units,), 1, dtype=torch.float64)
        self.input_dim=input_dim
        self.regularizer = kernel_regularizer
        self.initializer = kernel_initializer
        self.multiplier = 1
        self.mul = 1
        self.bias = False
    
    def build(self, input_dim, kernel : torch.Tensor = None, bias : torch.Tensor = None):
        # Ensure input_dim is defined properly if not passed.
        if input_dim[-1] is None:
            input_dim = (None, self.input_dim)
        else:
            self.input_dim = input_dim
        # Create kernel weights and D_i.
        if kernel is not None:
            if bias is None:
                self.W = kernel.clone()
                self.kernel = nn.Parameter(kernel.clone())
            else:
                self.W = kernel.clone()
                self.B = bias.clone().unsqueeze(0)
                self.kernel = nn.Parameter(torch.concat((kernel.clone(),bias.clone().unsqueeze(0))))
                self.bias = True
        else:
            self.kernel = nn.Parameter(torch.empty(input_dim[-1], self.units))
        self.D_i = nn.Parameter(torch.zeros(self.units))

        # Apply the initializer if provided.
        if self.initializer:
            self.kernel = self.initializer(self.kernel) # tu zmiana TODO

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1 ):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        
        if self.bias:
            max_W = torch.concat((torch.maximum(self.W * in_scalar,torch.zeros(self.kernel[:-1].shape)), self.B))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel*in_scalar,torch.zeros(self.kernel.shape))
        output_val = F.relu(torch.matmul(max_input,max_W))
        
        if self.bias:
            max_W = torch.concat((torch.maximum(self.W * in_scalar,torch.zeros(self.kernel[:-1].shape)), torch.maximum(self.B, torch.zeros(self.B.shape))))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel*in_scalar,torch.zeros(self.kernel.shape))
        max_V = F.relu(torch.max(torch.matmul(max_input,max_W)))+eps_V

        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)
        
        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
        else:
            self.multiplier = (self.t_max - self.t_min)/(max(minimal_t_max-self.t_min,1.0/L))+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
        
        if self.bias:
            self.kernel = nn.Parameter(torch.concat((self.W.clone()*(in_scalar/(self.multiplier)),self.B.clone()/(self.multiplier))))
        else:
            self.kernel = nn.Parameter(self.W.clone()*(in_scalar/(self.multiplier)))

        if self.bias:
            max_W = torch.concat((torch.maximum(self.kernel[:-1], torch.zeros(self.kernel[:-1].shape)), self.kernel[-1].unsqueeze(0)))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape))
        output_val = F.relu(torch.matmul(max_input,max_W))

        if self.bias:
            max_W = torch.concat((torch.maximum(self.kernel[:-1], torch.zeros(self.kernel[:-1].shape)), torch.maximum(self.kernel[-1].unsqueeze(0), torch.zeros(self.kernel[-1].unsqueeze(0).shape))))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape))
        max_V = F.relu(torch.max(torch.matmul(max_input,max_W)))+eps_V/(self.multiplier)

        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        
        # Returning for function signature consistency
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val, self.multiplier
    
    def forward(self, tj):
        """
        Input spiking times `tj`, output spiking times `ti` or membrane potential value for the output layer.
        """
        # Call the custom spiking logic
        if self.bias:
            # print(tj.shape)
            new_tj = torch.concat((tj, torch.tensor([[(self.t_min - 1)]])), dim=1)
            output = call_spiking(new_tj, self.kernel, self.D_i, self.t_min, self.t_max, noise=self.noise)
        else:
            output = call_spiking(tj, self.kernel, self.D_i, self.t_min, self.t_max, noise=self.noise)
        # If this is the output layer, perform the special integration logic
        if self.outputLayer:
            # Compute weighted product
            W_mult_x = torch.matmul(self.t_min - tj, self.kernel)
            self.alpha = self.D_i / (self.t_min - self.t_min_prev)
            output = self.alpha * (self.t_min - self.t_min_prev) + W_mult_x

        return output

class SpikingConv2D(nn.Module):
    def __init__(self, filters, name, X_n=1, padding='same', kernel_size=(3,3), robustness_params=None, kernels = None, device = 'cuda:0', biases = None, stride=1):
        super(SpikingConv2D, self).__init__()
        self.stride = stride
        if robustness_params is None:
            robustness_params = {}
        self.filters = filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.B_n = (1 + 0.0) * X_n
        self.t_min_prev, self.t_min, self.t_max = 0, 0, 1
        self.w_min, self.w_max = -1.0, 1.0
        self.time_bits = robustness_params.get('time_bits', 1)
        self.weight_bits = robustness_params.get('weight_bits', 1) 
        self.noise = robustness_params.get('noise', 0.0)
        self.device = device
        # Initialize alpha as a tensor of ones
        self.alpha = nn.Parameter(torch.ones(filters, dtype=torch.float32))
        
        # Registering the kernel as a learnable parameter
        #TODO:
        if kernels is not None:
            self.kernel = nn.Parameter(kernels).to(device)
        else:
            self.kernel = nn.Parameter(torch.randn(filters, 1, kernel_size[0], kernel_size[1], dtype=torch.float32)).to(device)
        if biases is not None:
            self.B = biases.unsqueeze(1).to(self.device)
        else:
            self.B = nn.Parameter(torch.zeros(filters, 1, dtype=torch.float32)).to(self.device)

        # Placeholder for batch normalization parameters
        self.BN = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        self.BN_before_ReLU = nn.Parameter(torch.tensor([0], dtype=torch.float32), requires_grad=False)
        
        # Parameter for different thresholds
        self.D_i = nn.Parameter(torch.zeros(9, filters, dtype=torch.float32)).to(self.device)

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        max_W = torch.maximum(self.kernel*(in_scalar),torch.zeros(self.kernel.shape).to(self.device))
        
        max_input = (in_ranges_max.unsqueeze(-1).unsqueeze(-1)).to(self.device) * torch.ones(self.kernel.shape[1:]).to(self.device)

        if self.B is not None:
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))+self.B.squeeze(1)))
            max_values = F.relu(torch.sum(torch.mul(max_input,max_W),(1,2,3))+self.B.squeeze(1))
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))+torch.maximum(self.B.squeeze(1), torch.zeros(self.B.squeeze(1).shape))))+eps_V
        else:
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))))+eps_V
            max_values = F.relu(torch.sum(torch.mul(max_input,max_W),(1,2,3)))
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)
        
        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
            
        else:
            if minimal_t_max-self.t_min==0:
                self.multiplier = (self.t_max - self.t_min)+eps
            else:
                self.multiplier = (self.t_max - self.t_min)/(max(minimal_t_max-self.t_min,1.0/L))+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
        self.kernel_mul = (in_scalar/(self.multiplier))

        max_W = torch.maximum(self.kernel*self.kernel_mul,torch.zeros(self.kernel.shape).to(self.device))
        
        max_input = (in_ranges_max.unsqueeze(-1).unsqueeze(-1)).to(self.device) * torch.ones(self.kernel.shape[1:]).to(self.device)

        if self.B is not None:
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))+self.B.squeeze(1)/self.multiplier))
            max_values = F.relu(torch.sum(torch.mul(max_input,max_W),(1,2,3))+self.B.squeeze(1)/self.multiplier)
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))+torch.maximum(self.B.squeeze(1), torch.zeros(self.B.squeeze(1).shape))/self.multiplier))+eps_V/(self.multiplier)
        else:
            max_V = F.relu(torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3))))+eps_V/(self.multiplier)
            max_values = F.relu(torch.sum(torch.mul(max_input,max_W),(1,2,3)))
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        # Returning for function signature consistency
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), max_values, self.multiplier

    def call_spiking(self, tj, W, D_i, t_min, t_max, noise):
        """
        Calculates spiking times from which ReLU functionality can be recovered.
        """
        threshold = t_max - t_min - D_i
        
        #### Check ####
        V = torch.matmul((tj - t_min), torch.maximum(W, torch.zeros(W.shape)))
        if((V>threshold).any()):
            print(f"ERROR SpikingConv2D V {V}, thr {threshold}") 
        ### END Check ###

        # Calculate output spiking time ti
        ti = torch.matmul(tj - t_min, W) + threshold + t_min
        
        # Ensure valid spiking time
        ti = torch.where(ti < t_max, ti, t_max)
        
        # Add noise
        if noise > 0:
            ti += torch.randn_like(ti) * noise
        
        return ti

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        if self.stride==1:
            padding_size = int(self.padding == 'same') * ((self.kernel_size[0]-1) // 2)
        else:
            # dont know if it works with stride other than 1 always set padding to valid
            padding_size = int(self.padding == 'same') * ((self.kernel_size[0]-1) // 2)
        image_same_size = tj.size(2) 
        image_valid_size = image_same_size - self.kernel_size[0] + 1


        tj_shape = tj.shape
        # Dodanie paddingu
        if self.padding == 'same':
            tj = torch.nn.functional.pad(tj, (padding_size, padding_size, padding_size, padding_size), value=self.t_min)
        elif type(self.padding) is tuple:
            tj = torch.nn.functional.pad(tj, (self.padding[0], self.padding[0], self.padding[1], self.padding[1]), value=self.t_min)
            pass
        # Wyciąganie patchy
        if self.stride==1:
            batch_size, in_channels, input_height, input_width = tj.shape
            tj = torch.nn.functional.unfold(tj, kernel_size=self.kernel_size, stride=1).transpose(1, 2)
            # Reshape dla wag
            W = self.kernel.view(self.filters, -1).t()
            out_channels, _, kernel_height, kernel_width = self.kernel.shape
            output_height = (input_height - kernel_height) // self.stride + 1
            output_width = (input_width - kernel_width) // self.stride + 1
        else:
            batch_size, in_channels, input_height, input_width = tj.shape
            tj = torch.nn.functional.unfold(tj, kernel_size=self.kernel_size, stride=self.stride).transpose(1, 2)
            out_channels, _, kernel_height, kernel_width = self.kernel.shape
            output_height = (input_height - kernel_height) // self.stride + 1
            output_width = (input_width - kernel_width) // self.stride + 1
            # Reshape dla wag
            W = self.kernel.view(out_channels, -1).t()
        
        
        
        if (self.padding == 'valid' or self.BN != 1 or self.BN_before_ReLU == 1) and (self.B is None): 

            ti = self.call_spiking(tj, W * self.kernel_mul, self.D_i[0], self.t_min, self.t_max, noise=self.noise).transpose(1, 2)
            if self.padding == 'valid':
                ti = ti.view(batch_size, out_channels, output_height, output_width)
            else:
                ti = ti.view(batch_size, out_channels, output_height, output_width)

        elif self.B is not None:
            ## concatenating simple "one" to vector of times
            one_as_time = self.t_min - 1
            tj = torch.concat((tj, one_as_time * torch.ones(tj.shape[0],tj.shape[1],1).to(self.device)), 2)
            ## conttenating biases to weight vector
            W = torch.concat((W * self.kernel_mul,self.B.T / self.multiplier),0)
            ti = self.call_spiking(tj, W, self.D_i[0], self.t_min, self.t_max, noise=self.noise).transpose(1, 2)
            if self.padding == 'valid':
                ti = ti.view(batch_size, out_channels, output_height, output_width)
            else:
                ti = ti.view(batch_size, out_channels, output_height, output_width)

        return ti

def fuse_conv_and_bn(conv, bn, device = 'cuda:0'):
	#
	# init
	fusedconv = torch.nn.Conv2d(
		conv.in_channels,
		conv.out_channels,
		kernel_size=conv.kernel_size,
		stride=conv.stride,
		padding=conv.padding,
		bias=True
	)
	#
	# prepare filters
	w_conv = conv.weight.clone().view(conv.out_channels, -1)
	w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps+bn.running_var)))
	with torch.no_grad():
		fusedconv.weight.copy_( torch.mm(w_bn, w_conv).view(fusedconv.weight.size()) )
	#
	# prepare spatial bias
	if conv.bias is not None:
		b_conv = conv.bias
	else:
		b_conv = torch.zeros( conv.weight.size(0) ).to(device)
	b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
	with torch.no_grad():
		fusedconv.bias.copy_( (torch.matmul(w_bn.to(device), b_conv.to(device)) + b_bn.to(device)) )
	
	return fusedconv.to(device)

import torch
import torch.nn as nn
import torch.nn.functional as F

class MaxMinPool2D(nn.Module):
    """
    Max Pooling or Min Pooling operation, depending on the sign of the batch normalization layer before.
    """
    def __init__(self, kernel_size, max_time, stride=None, padding=0, dilation=1):
        super(MaxMinPool2D, self).__init__()
        
        # Default sign is 1, indicating max pooling functionality.
        self.sign = nn.Parameter(-1*torch.ones(1, 1, 1, 1), requires_grad=False)
        self.dilation = dilation
        # MaxPool2d setup (will be used in call)
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

class AddSNNLayer(nn.Module):
    def __init__(self):
        super(AddSNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0, in_scalar1 = 1, in_scalar2 = 1):
        output_val = input1_val*in_scalar1 + input2_val*in_scalar2
        max_V = max(output_val)+eps_V
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)


        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
        else:
            self.multiplier = (self.t_max - self.t_min)/(max(minimal_t_max-self.t_min,1.0/L))+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
        self.mul1 = (in_scalar1/(self.multiplier))
        self.mul2 = (in_scalar2/(self.multiplier))

        output_val = input1_val*self.mul1 + input2_val*self.mul1
        max_V = max(output_val)+eps_V/(self.multiplier)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val, self.multiplier

    def forward(self, tj1, tj2):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i

        ### Check ###
        tj1_temp = (tj2-tj1)*(tj1<tj2)
        tj2_temp = (tj1-tj2)*(tj1>tj2)

        V = tj1_temp*self.mul1 + tj2_temp*self.mul2
        
        if((V>threshold).any()):
            print(f"ERROR AddSNNLayer1 V {V}, thr {threshold}")

        V = tj1*self.mul1 + tj2*self.mul2 - (self.mul1+self.mul2)*self.t_min
        if((V>threshold).any()):
            print(f"ERROR AddSNNLayer2 V {V}, thr {threshold}")
        
        ### END Check ###
        ti = tj1*self.mul1 + tj2*self.mul2 - (self.mul1+self.mul2)*self.t_min  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti

class SubSNNLayer(nn.Module):
    def __init__(self):
        super(SubSNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        self.multiplier = 1
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0, in_scalar1 = 1, in_scalar2 = 1):
        output_val = input1_val*in_scalar1
        self.input1_val = input1_val
        max_V = max(output_val)+eps_V
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)

        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
        else:
            self.multiplier = (self.t_max - self.t_min)/(max(minimal_t_max-self.t_min,1.0/L))+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
        self.mul1 = (in_scalar1/(self.multiplier))
        self.mul2 = (in_scalar2/(self.multiplier))

        output_val = input1_val*self.mul1
        max_V = max(output_val)+eps_V/(self.multiplier)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val, self.multiplier

    def forward(self, tj1, tj2):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        
        ### Check ###
        if(len(tj1.shape) == 3):
            if((torch.amax(self.t_min - tj1, dim=(1, 2))>(self.input1_val+eps_V)).any()):
                print(f"WARNING SubSNNLayer input times not in declared range mismach: {(torch.amax(self.t_min - tj1, dim=(1, 2))-self.input1_val).max()}")
        elif(len(tj1.shape) == 4):
            if((torch.amax(self.t_min - tj1, dim=(2, 3))>(self.input1_val+eps_V)).any()):
                print(f"WARNING SubSNNLayer input times not in declared range mismach: {(torch.amax(self.t_min - tj1, dim=(2, 3))-self.input1_val).max()}")
        tj1_temp = (tj2-tj1)*(tj1<tj2)
        tj2_temp = (tj1-tj2)*(tj1>=tj2)

        V = tj1_temp*self.mul1 - tj2_temp*self.mul2
        
        if((V>threshold).any()):
            print(f"ERROR SubSNNLayer1 V {V.max()}, thr {threshold}")
            print(f"{((tj1 - self.t_min)*self.mul1 - (tj2 - self.t_min)*self.mul2).max()}")

        V = (tj1 - self.t_min)*self.mul1 - (tj2 - self.t_min)*self.mul2
        if((V>threshold).any()):
            print(f"ERROR SubSNNLayer1 V {V.max()}, thr {threshold}")
        
        ### END Check ###

        ti = (tj1 - self.t_min)*self.mul1 - (tj2 - self.t_min)*self.mul2  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti

class IdentitySNNLayer(nn.Module):
    def __init__(self):
        super(IdentitySNNLayer, self).__init__()
        self.noise = 0
        self.B_n = 1
        pass

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        
        max_input = max(in_ranges_max*in_scalar)
        max_V = max_input+eps_V
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)

        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
        else:
            self.multiplier = (self.t_max - self.t_min)/(max(minimal_t_max-self.t_min,1.0/L))+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar,L)
        self.mul1 = (in_scalar/(self.multiplier))

        max_input = max(in_ranges_max*self.mul1)
        max_V = max_input+eps_V/(self.multiplier)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), in_ranges_max*self.mul1, self.multiplier

    def forward(self, tj):
        D_i = 0
        threshold = self.t_max - self.t_min - D_i

        V = (tj - self.t_min)*self.mul1
        
        if((V>threshold).any()):
            print(f"ERROR IdentitySNNLayer V {V.max()}, thr {threshold}")
        
        ti = (tj - self.t_min)*self.mul1  + threshold + self.t_min

        ti = torch.where(ti < self.t_max, ti, self.t_max)

        if self.noise > 0:
            ti += torch.randn_like(ti) * self.noise
        
        return ti
    
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

    def set_params(self, t_min_prev, t_min, input_val, minimal_t_max = 0, in_scalar = 1):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        t_min1, t_max1, conv1_val, in_scalar1 = self.conv1.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
        self.pooling1 = MaxMinPool2D(2, t_max1.data,2).to(self.device)
        t_min2, t_max2, conv2_val, in_scalar2 = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, in_ranges_max=conv1_val, in_scalar = in_scalar1)
        max_out2 = t_max2 - t_min2
        
        if self.downsample:
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val, in_scalar=in_scalar)
            self.pooling2 = MaxMinPool2D(2, t_max1_dummy.data,2).to(self.device)
        else:
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val, in_scalar=in_scalar)

        t_min2, t_max2, conv2_val, in_scalar2  = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, minimal_t_max=t_max1_dummy, in_ranges_max=conv1_val, in_scalar=in_scalar1)
        
        # time t_max2 and t_max1_dummy are the same
        t_min_add = t_max2 - max(max_dummy1, max_out2)

        self.t_min, self.t_max, add_val, out_scalar = self.add_layer.set_params(t_min_add, t_max2, conv2_val, downsample_val, in_scalar1=in_scalar2, in_scalar2=in_scalar_downsample)

        self.times = [(t_min1, t_max1, 'c'), (t_min2, t_max2, 'c'), (self.t_min, self.t_max, 'a') ]
        return self.t_min, self.t_max, add_val, out_scalar

    def get_main_times(self):
        return self.times

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

class LayerSNN(nn.Module):
    def __init__(self, layer, inplanes, planes, blocks, stride=1, device = 'cuda:0'):
        self.inplanes = inplanes

        downsample = None
        if stride != 1 or self.inplanes != planes:
            conv2d, bias_from_nn = layer[0].downsample[0], layer[0].downsample[1]
            conv_fused = fuse_conv_and_bn(conv2d, bias_from_nn)
            robustness_params={
                'noise':0.0,
                'time_bits':0,
                'weight_bits': 0,
                'latency_quantiles':0.0
            }
            downsample = SpikingConv2D(planes,"test2", padding='same', stride=1, device=device,robustness_params=robustness_params,kernels=conv_fused.weight.data, biases=conv_fused.bias, kernel_size=(1,1))
            # t_min, t_max = spiking_conv2.set_params(0,1)stride
            
        self.layers = []
        self.layers.append(ResidualSNNBlock(layer[0],self.inplanes,planes, 1, downsample=downsample, device=device))
        self.inplanes = planes
        self.blocks = blocks
        for i in range(1, blocks):
            self.layers.append(ResidualSNNBlock(layer[i],self.inplanes,planes, 1, downsample=None, device=device))
        
    
    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        tmin, tmax = t_min_prev, t_min
        for i in range(self.blocks):
            tmin, tmax, in_ranges_max, in_scalar = self.layers[i].set_params(tmin, tmax,in_ranges_max, in_scalar=in_scalar)
        return tmin, tmax, in_ranges_max, in_scalar
    
    def get_main_times(self):
        lst = []
        for i in range(self.blocks):
            lst.extend(self.layers[i].get_main_times())
        return lst
    
    def forward(self, x):
        for i in range(self.blocks):
            x = self.layers[i].forward(x)
        return x

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

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        if(in_ranges_max.shape[0] != self.filters):
            in_ranges_max = torch.concat((in_ranges_max,torch.zeros(in_ranges_max.shape)))
        tmin1, tmax1, first_val, in_scalar1 = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max, in_scalar=in_scalar)
        tmin2, tmax2, second_val, in_scalar2 = self.conv_second.set_params(t_min_prev, t_min, in_ranges_max, tmax1, in_scalar=in_scalar)

        tmin1, tmax1, first_val, in_scalar1 = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max, tmax2, in_scalar=in_scalar)
        self.t_max = tmax1
        tmins, tmaxs, sub_val, in_scalar_sub = self.sub.set_params(t_min, tmax1, first_val,second_val, in_scalar1=in_scalar1, in_scalar2=in_scalar2)
        tmaxs = max(min(tmaxs,tmins+1.0/in_scalar_sub)+eps_V, minimal_t_max)
        # tmaxs = max(tmaxs, minimal_t_max)
        self.sub.t_max = tmaxs
        self.t_max = tmaxs
        # Returning for function signature consistency
        return tmins, self.t_max, torch.minimum(sub_val, torch.ones(sub_val.shape)/in_scalar_sub), in_scalar_sub
        # return tmin1, self.t_max, torch.minimum(first_val, torch.ones(first_val.shape))

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        tj1 = self.conv_first(tj)
        tj2 = self.conv_second(tj)
        tj_sub = self.sub(tj1, tj2)

        return tj_sub

class AddSNNLayer_all(nn.Module):
    """
    """
    def __init__(self, bias=0):
        super(AddSNNLayer_all, self).__init__()
        self.noise = 0
        self.B_n = 1
        self.B = bias # bias for all inputs (for Hard tanh)
        pass

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0, in_scalar1 = 1, in_scalar2 = 1):
        self.input1_val = input1_val
        if input2_val.shape[0] != input1_val.shape[0]:
            input2_val = torch.concat((input2_val, torch.zeros(input2_val.shape)))
        output_val = input1_val*in_scalar1 + input2_val*in_scalar2 + self.B
        max_V = max(output_val)+eps_V
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)

        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
            
        else:
            self.multiplier = (self.t_max - self.t_min + eps)/(max(minimal_t_max-self.t_min,1.0/L))
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)

        #### adding epsilon block
        self.multiplier_temp = self.multiplier
        output_val = input1_val*in_scalar1 + input2_val*in_scalar2 + self.B
        max_V = max(output_val)+eps_V*self.multiplier_temp
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)

        if(minimal_t_max == 0):
            self.multiplier = self.t_max - self.t_min
            self.multiplier = self.multiplier*L+eps
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
            
        else:
            self.multiplier = (self.t_max - self.t_min + eps)/(max(minimal_t_max-self.t_min,1.0/L))
            if self.multiplier==0:
                self.multiplier=max(L*in_scalar1,L,L*in_scalar2)
        #### adding epsilon block
        self.mul1 = (in_scalar1/(self.multiplier))
        self.mul2 = (in_scalar2/(self.multiplier))

        output_val = input1_val*self.mul1 + input2_val*self.mul2 + self.B/self.multiplier
        max_V = max(output_val)+eps_V*self.multiplier_temp/(self.multiplier)
        # print(f"epsilon: {eps_V*self.multiplier_temp/(self.multiplier)}, mul: {self.multiplier}")
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)
        
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val, self.multiplier

    def forward(self, tj1, tj2):

        self.channels = tj1.shape[1]//2

        D_i = 0
        threshold = self.t_max - self.t_min - D_i
        #### Check ####
        if(len(tj1.shape) == 3):
            if((torch.amax(self.t_min - tj1, dim=(1, 2))>(self.input1_val+eps_V)).any()):
                print(f"WARNING SubSNNLayer input times not in declared range mismach: {(torch.amax(self.t_min - tj1, dim=(1, 2))-self.input1_val).max()}")
        elif(len(tj1.shape) == 4):
            if((torch.amax(self.t_min - tj1, dim=(2, 3))>(self.input1_val+eps_V)).any()):
                print(f"WARNING SubSNNLayer input times not in declared range mismach: {(torch.amax(self.t_min - tj1, dim=(2, 3))-self.input1_val).max()}")
        tj1_temp = tj1[0, :self.channels].detach().clone()
        tj2_temp = tj1[0, self.channels:].detach().clone()
        tj3_temp = tj2[0, :self.channels].detach().clone()
        tj4_temp = tj2[0, self.channels:].detach().clone()

        stacked = torch.stack([tj1_temp, tj2_temp, tj3_temp, tj4_temp], dim=0)

        min_times_list = []
        mask_list = []

        for i in range(4):
            min_indices = torch.argmin(stacked, dim=0)


            min_vals = stacked.gather(0, min_indices.unsqueeze(0)).squeeze(0)
            min_times_list.append(min_vals)

            mask = torch.zeros_like(stacked, dtype=torch.bool)
            C, X, Y = min_indices.shape
            c_idx, x_idx, y_idx = torch.meshgrid(
                torch.arange(C), torch.arange(X), torch.arange(Y), indexing="ij"
            )
            mask[min_indices, c_idx, x_idx, y_idx] = True
            if i!=0:
                mask_list.append(torch.logical_or(mask,mask_list[-1]))
            else:
                mask_list.append(mask)
            stacked = torch.where(mask, torch.full_like(stacked, float('inf')), stacked)
        min_times_list.append(self.t_max*torch.ones(min_times_list[0].shape))

        min_times = torch.stack(min_times_list, dim=0)
        mask_list = torch.stack(mask_list, dim=0)
        vect = -torch.tensor([self.mul1,-self.mul1, self.mul2, -self.mul2])
        V_plus = torch.zeros(min_times[0].shape) + (0 if self.t_max==self.t_min else ((self.B/(self.multiplier * (self.t_max-self.t_min)))* (min_times[0]-self.t_min)))
        V_minus = torch.zeros(min_times[0].shape) + (0 if self.t_max==self.t_min else ((self.B/(self.multiplier * (self.t_max-self.t_min)))* (min_times[0]-self.t_min)))
        for i in range(4):
            duration = min_times[i+1] - min_times[i]
            if i!=0:
                mask_list[i] =  torch.logical_or(mask_list[i], mask_list[i-1])
            V_plus += ((mask_list[i].to(torch.float64).T)@vect).T * duration + ( 0 if self.t_max==self.t_min else ((self.B/(self.multiplier * (self.t_max-self.t_min)))* (min_times[i+1]-min_times[i])) )
            if((V_plus>threshold).any()):
                print(f"ERROR AddSNNLayer_all V {V_plus.max()}, thr {threshold}")
            V_minus += (-((mask_list[i].to(torch.float64).T)@vect).T * duration) + ( 0 if self.t_max==self.t_min else ((self.B/(self.multiplier * (self.t_max-self.t_min)))* (min_times[i+1]-min_times[i])) )
            if((V_minus>threshold).any()):
                print(f"ERROR AddSNNLayer_all V {V_minus.max()}, thr {threshold}")
        V2 = (tj1[0, :self.channels] - tj1[0, self.channels:])*self.mul1 + (tj2[0, :self.channels] - tj2[0, self.channels:])*self.mul2 + self.B/(self.multiplier)*(1)
        if ((V2 - V_plus).abs().max() > 0.00001):
            print(f"WARNING AddSNNLayer_all too big difference in chceck: {(V2 - V_plus).abs().max()}")
        V2 = (tj1[0, self.channels:] - tj1[0, :self.channels])*self.mul1 + (tj2[0, self.channels:] - tj2[0, :self.channels])*self.mul2 + self.B/(self.multiplier)*(1)
        if ((V2 - V_minus).abs().max() > 0.00001):
            print(f"WARNING AddSNNLayer_all too big difference in chceck: {(V2 - V_minus).abs().max()}")
        

        #### END Check ####
        ti = torch.concat(((tj1[0, :self.channels]- tj1[0, self.channels:])*self.mul1 + (tj2[0, :self.channels]  - tj2[0, self.channels:])*self.mul2, 
                           (tj1[0, self.channels:] - tj1[0, :self.channels])*self.mul1 + (tj2[0, self.channels:] - tj2[0, :self.channels])*self.mul2)) + self.B/(self.multiplier)*(1) + threshold + self.t_min

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

    def set_params(self, t_min_prev, t_min, input1_val, input2_val, minimal_t_max = 0, in_scalar1 = 1, in_scalar2 = 1):
        tmin1, tmax1, first_val, in_scalar_first = self.first.set_params(t_min_prev, t_min,input1_val, input2_val, in_scalar1=in_scalar1, in_scalar2=in_scalar2)
        tmin2, tmax2, second_val, in_scalar_second = self.second.set_params(t_min_prev, t_min,input1_val, input2_val, minimal_t_max=tmax1, in_scalar1=in_scalar1, in_scalar2=in_scalar2)

        tmin1, tmax1, first_val, in_scalar_first = self.first.set_params(t_min_prev, t_min,input1_val, input2_val, minimal_t_max=tmax2, in_scalar1=in_scalar1, in_scalar2=in_scalar2)

        tmins, tmaxs, sub_val, in_scalar_sub = self.sub.set_params(t_min, tmax1, first_val,second_val, in_scalar1=in_scalar_first, in_scalar2=in_scalar_second) ## t_min as angument do nothing
        
        self.sub.t_max = max(tmaxs, minimal_t_max)
        return tmins, max(tmaxs, minimal_t_max), torch.minimum(sub_val,(1.0/in_scalar_sub)), in_scalar_sub

    def forward(self, tj1, tj2):
        tj_first = self.first(tj1, tj2)
        tj_second = self.second(tj1, tj2)
        tj_sub = self.sub(tj_first, tj_second)
        return tj_sub
    
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

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        tmin1, tmax1, first_val, out_scalar = self.conv_first.set_params(t_min_prev, t_min, in_ranges_max, in_scalar = in_scalar)
        
        # Returning for function signature consistency
        return tmin1, tmax1, first_val, out_scalar
        # return tmin1, self.t_max, torch.minimum(first_val, torch.ones(first_val.shape))

    def forward(self, tj):
        """
        Input spiking times tj, output spiking times ti. 
        """
        tj1 = self.conv_first(tj)
        return tj1

class ResidualSNNBlock_all(nn.Module):
    def __init__(self,resblock : ResidualBlock, in_channels, out_channels, stride=1, downsample=None, robustness_params = None, device = "cuda:0", end_maxpool = False):
        super(ResidualSNNBlock_all, self).__init__()
        conv = resblock.conv1[0]
        bn= resblock.conv1[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn)
        if (downsample is not None):
            self.conv1 = SpikingConv2D_all(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)
        else:
            self.conv1 = SpikingConv2D_Htanh(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)
        self.device = device
        
        conv = resblock.conv2[0]
        bn= resblock.conv2[1]
        bn.eval()
        conv_fused = fuse_conv_and_bn(conv, bn)
        self.conv2 = SpikingConv2D_Htanh(out_channels, "temp1", device=device, padding=(1,1), stride=stride, kernel_size=(3,3),robustness_params=robustness_params, kernels=conv_fused.weight.data, biases= conv_fused.bias.data)

        self.downsample = downsample
        self.identity = IdentitySNNLayer()
        if end_maxpool:
            self.add_layer = AddSNNLayer_all()
        else:
            self.add_layer = AddSNNLayer_Htanh()
        self.out_channels = out_channels
        self.end_maxpool = end_maxpool

    def set_params(self, t_min_prev, t_min, input_val, minimal_t_max = 0, in_scalar = 1):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        t_min1, t_max1, conv1_val, in_scalar1 = self.conv1.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
        self.t_max1 = t_max1
        self.pooling1 = MaxMinPool2D(2, t_max1.data,2).to(self.device)
        t_min2, t_max2, conv2_val, in_scalar2 = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, in_ranges_max=conv1_val, in_scalar=in_scalar1)
        max_out2 = t_max2 - t_min2
        
        if self.downsample:
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.downsample.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val, in_scalar=in_scalar)
            self.t_max1_dummy = t_max1_dummy
            self.pooling2 = MaxMinPool2D(2, t_max1_dummy.data,2).to(self.device)
        else:
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min, in_ranges_max=input_val, in_scalar=in_scalar)
            max_dummy1 = t_max1_dummy - t_min_dummy
            t_min_dummy, t_max1_dummy, downsample_val, in_scalar_downsample = self.identity.set_params(t_min_prev=t_min_prev,t_min=t_min,minimal_t_max=t_max2, in_ranges_max=input_val, in_scalar=in_scalar)

        t_min2, t_max2, conv2_val, in_scalar2 = self.conv2.set_params(t_min_prev=t_min1,t_min=t_max1, minimal_t_max=t_max1_dummy, in_ranges_max=conv1_val, in_scalar=in_scalar1)
        
        # time t_max2 and t_max1_dummy are the same
        t_min_add = t_max2 - max(max_dummy1, max_out2)

        self.t_min, self.t_max, add_val, out_scalar = self.add_layer.set_params(t_min_add, t_max2, conv2_val, downsample_val, in_scalar1=in_scalar2, in_scalar2=in_scalar_downsample)

        self.times = [(t_min1, t_max1, 'c'), (t_min2, t_max2, 'c'), (self.t_min, self.t_max, 'a') ]
        return self.t_min, self.t_max, add_val, out_scalar

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
            conv_fused = fuse_conv_and_bn(conv2d, bias_from_nn)
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
            
        
    
    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0, in_scalar = 1):
        tmin, tmax = t_min_prev, t_min
        for i in range(self.blocks):
            tmin, tmax, in_ranges_max, in_scalar = self.layers[i].set_params(tmin, tmax,in_ranges_max, in_scalar=in_scalar)
        return tmin, tmax, in_ranges_max, in_scalar
    
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

class SpikingDense_positive_tanH(nn.Module):
    """
    from positive to all and Hard tanh
    """
    def __init__(self, units, input_units, name, weights, biases, X_n=1, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        super().__init__()
        W = torch.concatenate((weights.T, -weights.T),dim=1)
        b1 = torch.concatenate((biases, -biases))
        b2 = torch.concatenate((biases, -biases)) - 1
        self.first = SpikingDense(2*units,"test",robustness_params=robustness_params)
        self.second = SpikingDense(2*units,"test",robustness_params=robustness_params)
        self.first.build((input_units,), W.detach().clone(), b1)
        self.second.build((input_units,), W.detach().clone(), b2)
        self.sub_layer = SubSNNLayer()



    def set_params(self, t_min_prev, t_min, in_ranges_max, in_scalar=1):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        tmin1, tmax1, first_val, in_scalar1 = self.first.set_params(t_min_prev, t_min,in_ranges_max, in_scalar=in_scalar)
        tmin2, tmax2, second_val, in_scalar2 = self.second.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=tmax1, in_scalar=in_scalar)

        tmin1, tmax1, first_val, in_scalar1 = self.first.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=tmax2, in_scalar=in_scalar)

        tmins, tmaxs, sub_val, in_scalar_sub = self.sub_layer.set_params(t_min, tmax1, first_val,second_val, in_scalar1=in_scalar1, in_scalar2=in_scalar2) ## t_min as angument do nothing
        self.sub_layer.t_max = min(tmaxs,tmins+(1/in_scalar_sub))+eps_V
        return tmins, min(tmaxs,tmins+(1/in_scalar_sub))+eps_V, torch.maximum(sub_val, 1/in_scalar_sub), in_scalar_sub
    
    def forward(self, tj):
        """
        Input spiking times `tj`, output spiking times `ti` or membrane potential value for the output layer.
        """
        # Call the custom spiking logic
        out1 = self.first(tj)
        out2 = self.second(tj)
        sub_ = self.sub_layer(out1,out2)
        return sub_

class SpikingDense_all_all(nn.Module):
    """
    from all to all (pure linear layer)
    """
    def __init__(self, units, input_units, name, weights, biases, X_n=1, outputLayer=False, robustness_params={}, input_dim=None,
                 kernel_regularizer=None, kernel_initializer=None):
        super().__init__()
        W1 = torch.concatenate((weights.T, -weights.T),dim=1)
        W2 = torch.concatenate((-weights.T, weights.T),dim=1)
        W = torch.concatenate((W1,W2),dim=0)
        b1 = torch.concatenate((biases, -biases))
        self.first = SpikingDense(2*units,"test",robustness_params=robustness_params)
        self.first.build((2*input_units,), W, b1)



    def set_params(self, t_min_prev, t_min, in_ranges_max, in_scalar=1):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        tmin1, tmax1, first_val, in_scalar = self.first.set_params(t_min_prev, t_min,in_ranges_max, in_scalar=in_scalar)
        return tmin1, tmax1, first_val, in_scalar
    
    def forward(self, tj):
        """
        Input spiking times `tj`, output spiking times `ti` or membrane potential value for the output layer.
        """
        # Call the custom spiking logic
        out1 = self.first(tj)
        return out1
