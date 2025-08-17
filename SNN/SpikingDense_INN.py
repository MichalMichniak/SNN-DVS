import torch
import torch.nn as nn
import torch.nn.functional as F
from SNN.SubSNNLayer import SubSNNLayer

def call_spiking(tj, W, D_i, t_min, t_max, noise, dtype=torch.FloatTensor):
    """
    Calculates spiking times to recover ReLU-like functionality.
    Assumes tau_c=1 and B_i^(n)=1.
    """
    # Calculate the spiking threshold (Eq. 18)
    threshold = t_max - t_min - D_i
    
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

    def set_params(self, t_min_prev, t_min, in_ranges_max, minimal_t_max = 0):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        
        if self.bias:
            max_W = torch.concat((torch.maximum(self.kernel[:-1],torch.zeros(self.kernel[:-1].shape)), self.kernel[-1].unsqueeze(0)))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape))
        output_val = F.relu(torch.matmul(max_input,max_W))

        if self.bias:
            max_W = torch.concat((torch.maximum(self.kernel[:-1],torch.zeros(self.kernel[:-1].shape)), torch.maximum(self.kernel[-1].unsqueeze(0),torch.zeros(self.kernel[-1].unsqueeze(0).shape))))
            max_input = torch.concat((torch.tensor(in_ranges_max), torch.tensor([(1)])))
        else:
            max_input = torch.tensor(in_ranges_max)
            max_W = torch.maximum(self.kernel,torch.zeros(self.kernel.shape))
        max_V = F.relu(torch.max(torch.matmul(max_input,max_W)))

        self.t_min_prev = torch.tensor(t_min_prev, dtype=torch.float64, requires_grad=False)
        self.t_min = torch.tensor(t_min, dtype=torch.float64, requires_grad=False)
        self.t_max = torch.tensor(max(t_min + self.B_n*max_V, minimal_t_max), dtype=torch.float64, requires_grad=False)

        
        # Returning for function signature consistency
        return t_min, max(t_min + self.B_n*max_V, minimal_t_max), output_val
    
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
        self.first.build((input_units,), W, b1)
        self.second.build((input_units,), W, b2)
        self.sub_layer = SubSNNLayer()



    def set_params(self, t_min_prev, t_min, in_ranges_max):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        tmin1, tmax1, first_val = self.first.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=t_min+1)
        tmin2, tmax2, second_val = self.second.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=tmax1)

        tmin1, tmax1, first_val = self.first.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=tmax2)

        tmins, tmaxs, sub_val = self.sub_layer.set_params(t_min, tmax1, first_val,second_val) ## t_min as angument do nothing
        self.sub_layer.t_max = tmins+1
        return tmins, tmins+1, torch.minimum(sub_val, torch.ones(sub_val.shape))
    
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



    def set_params(self, t_min_prev, t_min, in_ranges_max):
        """
        Set t_min_prev, t_min, t_max parameters of this layer. Alpha is fixed at 1.
        """
        tmin1, tmax1, first_val = self.first.set_params(t_min_prev, t_min,in_ranges_max, minimal_t_max=t_min+1)
        return tmin1, tmax1, first_val
    
    def forward(self, tj):
        """
        Input spiking times `tj`, output spiking times `ti` or membrane potential value for the output layer.
        """
        # Call the custom spiking logic
        out1 = self.first(tj)
        return out1