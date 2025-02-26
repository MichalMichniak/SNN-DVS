import torch
import torch.nn as nn

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

    def set_params(self, t_min_prev, t_min, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max, J_ij (kernel) and vartheta_i (threshold) parameters of this layer.
        """
        max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape).to(self.device))
        max_input = (t_min - t_min_prev) * torch.ones(self.kernel.shape).to(self.device)
        if self.B is not None:
            max_V = torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3)) + self.B.unsqueeze(dim=1))
        else:
            max_V = torch.max(torch.sum(torch.mul(max_input,max_W),(1,2,3)))
        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        
        # Returning for function signature consistency
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max)

    def call_spiking(self, tj, W, D_i, t_min, t_max, noise):
        """
        Calculates spiking times from which ReLU functionality can be recovered.
        """
        threshold = t_max - t_min - D_i
        
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
            # tj = tj.view(-1, W.size(0))
            ti = self.call_spiking(tj, W, self.D_i[0], self.t_min, self.t_max, noise=self.noise).transpose(1, 2)
            if self.padding == 'valid':
                # ti = ti.view(-1, image_valid_size, image_valid_size, self.filters)
                ti = ti.view(batch_size, out_channels, output_height, output_width)
                #ti = torch.nn.functional.fold(ti, (tj_shape[-1],tj_shape[-1]), (1, 1)) #assuming square input
            else:
                ti = ti.view(batch_size, out_channels, output_height, output_width)
                #ti = torch.nn.functional.fold(ti, (tj_shape[-1],tj_shape[-1]), (1, 1)) #assuming square input
                # ti = ti.view(-1, image_same_size, image_same_size, self.filters)
        elif self.B is not None:
            ## concatenating simple "one" to vector of times
            one_as_time = self.t_min - 1
            tj = torch.concat((tj, one_as_time * torch.ones(tj.shape[0],tj.shape[1],1).to(self.device)), 2)
            ## conttenating biases to weight vector
            W = torch.concat((W,self.B.T),0)
            ti = self.call_spiking(tj, W, self.D_i[0], self.t_min, self.t_max, noise=self.noise).transpose(1, 2)
            if self.padding == 'valid':
                # ti = ti.view(-1, image_valid_size, image_valid_size, self.filters)
                # ti = torch.nn.functional.fold(ti, (tj_shape[-1],tj_shape[-1]), (1, 1)) #assuming square input
                ti = ti.view(batch_size, out_channels, output_height, output_width)
            else:
                # ti = torch.nn.functional.fold(ti, (tj_shape[-1],tj_shape[-1]), (1, 1)) #assuming square input
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
		fusedconv.bias.copy_( (torch.matmul(w_bn, b_conv) + b_bn) )
	
	return fusedconv.to(device)