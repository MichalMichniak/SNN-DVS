import pytest
from SNN.SpikingConv2D import SpikingConv2D
import torch
import torch.nn as nn

testdata = [
    (8,3,4,8,120),
    (16,64,20,120,120),
    (128,32,1800,2000,120),
    (128,32,2000,2200,120),
    ]

@pytest.mark.parametrize("input_, output_, t_min_prev, t_max_prev, seed", testdata)
def test_Dense_numerical_error(input_, output_, t_min_prev, t_max_prev, seed):
    torch.manual_seed(seed)
    robustness_params={
    'noise':0.0,
    'time_bits':0,
    'weight_bits': 0,
    'latency_quantiles':0.0
    }
    torch.manual_seed(seed)
    random_tensor = torch.rand(input_)

    spiking_dense = SpikingDense(output_,"test",robustness_params=robustness_params)
    weights = torch.rand(input_,output_,dtype=torch.float32)
    spiking_dense.build((input_,),weights)
    t_min, t_max = spiking_dense.set_params(t_min_prev, t_max_prev)

    t_input = t_max_prev - random_tensor
    inputs= t_max_prev - t_input
    linear_troch = torch.nn.Linear(input_,output_, bias=False)
    linear_troch.weight = nn.parameter.Parameter(weights.T)
    gtruth = linear_troch(inputs)

    assert (spiking_dense.t_max - spiking_dense(t_input) - gtruth).abs().max() < 0.001


@pytest.mark.parametrize("input_, output_, t_min_prev, t_max_prev, seed", testdata)
def test_Dense_time_constrains(input_, output_, t_min_prev, t_max_prev, seed):
    torch.manual_seed(seed)
    robustness_params={
    'noise':0.0,
    'time_bits':0,
    'weight_bits': 0,
    'latency_quantiles':0.0
    }
    torch.manual_seed(seed)
    random_tensor = torch.rand(input_)

    spiking_dense = SpikingDense(output_,"test",robustness_params=robustness_params)
    weights = torch.rand(input_,output_,dtype=torch.float32)
    spiking_dense.build((input_,),weights)
    t_min, t_max = spiking_dense.set_params(t_min_prev, t_max_prev)

    t_input = t_max_prev - random_tensor
    inputs= t_max_prev - t_input
    linear_troch = torch.nn.Linear(input_,output_, bias=False)
    linear_troch.weight = nn.parameter.Parameter(weights.T)
    gtruth = linear_troch(inputs)


    assert spiking_dense.t_min == t_max_prev 
    assert spiking_dense.t_max>t_max_prev 
    assert (spiking_dense(t_input)<=spiking_dense.t_max).all() 
    assert (spiking_dense(t_input)>=t_max_prev).all()