import math
import needle as ndl



def rand(*shape, low=0.0, high=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random numbers uniform between low and high """
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) * (high - low) + low
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)
    

def randn(*shape, mean=0.0, std=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate random normal with specified mean and std deviation """
    device = ndl.cpu() if device is None else device
    array = device.randn(*shape) * std + mean
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def constant(*shape, c=1.0, device=None, dtype="float32", requires_grad=False):
    """ Generate constant Tensor """
    device = ndl.cpu() if device is None else device
    array = device.ones(*shape, dtype=dtype) * c # note: can change dtype
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def ones(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-ones Tensor """
    return constant(*shape, c=1.0, device=device, dtype=dtype, requires_grad=requires_grad)


def zeros(*shape, device=None, dtype="float32", requires_grad=False):
    """ Generate all-zeros Tensor """
    return constant(*shape, c=0.0, device=device, dtype=dtype, requires_grad=requires_grad)


def randb(*shape, p=0.5, device=None, dtype="bool", requires_grad=False):
    """ Generate binary random Tensor """
    device = ndl.cpu() if device is None else device
    array = device.rand(*shape) <= p
    return ndl.Tensor(array, device=device, dtype=dtype, requires_grad=requires_grad)


def one_hot(n, i, device=None, dtype="float32", requires_grad=False):
    """ Generate one-hot encoding Tensor """
    device = ndl.cpu() if device is None else device
    return ndl.Tensor(device.one_hot(n,i.numpy(), dtype=dtype), device=device, requires_grad=requires_grad)


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = gain * math.sqrt(6 / (fan_in + fan_out))
    return rand(fan_in, fan_out, low=-a, high=a, **kwargs)
    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = gain * math.sqrt(2 / (fan_in + fan_out))
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION


def kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    bound = math.sqrt(6 / fan_in)
    return rand(fan_in, fan_out, low=-bound, high=bound, **kwargs)
    ### END YOUR SOLUTION


def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2 / fan_in)
    return randn(fan_in, fan_out, mean=0, std=std, **kwargs)
    ### END YOUR SOLUTION
