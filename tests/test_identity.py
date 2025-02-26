import pytest
from SNN.Identity import IdentitySNNLayer
import torch

testdata = [
    (8,3,64,64,4,8,120),
    (16,64,64,64,20,120,120),
    (128,32,32,32,2000,2500,120),
    (128,32,32,32,20000,25000,120),
    ]

@pytest.mark.parametrize("batch, channel, width, height, t_min_prev, t_max_prev, seed", testdata)
def test_Identity_numerical_error(batch, channel, width, height, t_min_prev, t_max_prev, seed):
    torch.manual_seed(seed)
    random_tensor = torch.rand(batch, channel, width, height)

    

    input1 = t_max_prev - random_tensor

    idlayer = IdentitySNNLayer()
    idlayer.set_params(t_min_prev, t_max_prev,2)
    result = idlayer(input1)
    gtruth = random_tensor


    assert (idlayer.t_max-result - gtruth).abs().max() < 0.001


@pytest.mark.parametrize("batch, channel, width, height, t_min_prev, t_max_prev, seed", testdata)
def test_Identity_time_constrains(batch, channel, width, height, t_min_prev, t_max_prev, seed):
    torch.manual_seed(seed)
    random_tensor = torch.rand(batch, channel, width, height)

    

    input1 = t_max_prev - random_tensor

    idlayer = IdentitySNNLayer()
    idlayer.set_params(t_min_prev, t_max_prev,2)
    result = idlayer(input1)
    gtruth = random_tensor


    assert idlayer.t_min == t_max_prev 
    assert idlayer.t_max>t_max_prev 
    assert (result<=idlayer.t_max).all() 
    assert (result>=t_max_prev).all()