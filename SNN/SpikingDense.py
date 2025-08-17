import torch
import torch.nn as nn

def call_spiking(tj, W, D_i, t_min, t_max, noise):
    """
    Calculates spiking times to recover ReLU-like functionality.
    Assumes tau_c=1 and B_i^(n)=1.
    """
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    
    # Calculate output spiking time ti (Eq. 7)

    ti = torch.matmul(tj - t_min, W) + threshold + t_min
    
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
        self.B_n = (1 + 0.5) * X_n
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
                self.kernel = nn.Parameter(kernel.clone())
            else:
                self.kernel = nn.Parameter(torch.concat((kernel.clone(),bias.clone().unsqueeze(0))))
                self.bias = True
        else:
            self.kernel = nn.Parameter(torch.empty(input_dim[-1], self.units))
        self.D_i = nn.Parameter(torch.zeros(self.units))

        # Apply the initializer if provided.
        if self.initializer:
            self.kernel = self.initializer(self.kernel) # tu zmiana TODO

    def set_params(self, t_min_prev, t_min):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape))
        if self.bias:
            max_input = torch.concat(((t_min - t_min_prev) * torch.ones(self.input_dim), torch.tensor([(t_min - 1)])))
        else:
            max_input = (t_min - t_min_prev) * torch.ones(self.input_dim)
        max_V = torch.max(torch.matmul(max_input,max_W))

        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(t_min + self.B_n*max_V, dtype=torch.float64, requires_grad=False)

        
        # Returning for function signature consistency
        return t_min, t_min + self.B_n*max_V
    
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