"""This file defies specific implementations of devices when using numpy as NDArray backend.
"""
import numpy


class Device:
    """Baseclass of all device"""


class CPUDevice(Device):
    """Represents data that sits in CPU"""

    def __repr__(self):
        return "needle.cpu()"

    def __hash__(self):
        return self.__repr__().__hash__()

    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def randn(self, *shape, dtype="float32"):
        return numpy.random.randn(*shape).astype(dtype)

    def rand(self, *shape, dtype="float32"):
        return numpy.random.rand(*shape).astype(dtype)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]

    def empty(self, shape, dtype="float32"):
        return numpy.empty(shape, dtype=dtype)

    def full(self, shape, fill_value, dtype="float32"):
        return numpy.full(shape, fill_value, dtype=dtype)


def cpu():
    """Return cpu device"""
    return CPUDevice()


def default_device():
    return cpu()


def all_devices():
    """return a list of all available devices"""
    return [cpu()]
