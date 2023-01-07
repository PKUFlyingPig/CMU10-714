import sys
sys.path.append('./python')
sys.path.append('./apps')
from simple_ml import *
import numdifftools as nd

import numpy as np
import mugrade
import needle as ndl


##############################################################################
### TESTS/SUBMISSION CODE FOR forward passes
def test_divide_forward():
    np.testing.assert_allclose(ndl.divide(ndl.Tensor([[3.3 , 4.35, 1.2 ],
       [2.45, 0.95, 2.55]]), ndl.Tensor([[4.6 , 4.35, 4.8 ],
       [0.65, 0.7 , 4.4 ]])).numpy(), np.array([[0.717391304348, 1.            , 0.25          ],
       [3.769230769231, 1.357142857143, 0.579545454545]]))


def test_divide_scalar_forward():
    np.testing.assert_allclose(ndl.divide_scalar(ndl.Tensor([[1.7 , 1.45]]), scalar=12).numpy(), 
        np.array([[0.141666666667, 0.120833333333]]))


def test_matmul_forward():
    np.testing.assert_allclose(ndl.matmul(ndl.Tensor([[4.95, 1.75, 0.25],
       [4.15, 4.25, 0.3 ],
       [0.3 , 0.4 , 2.1 ]]), ndl.Tensor([[1.35, 2.2 , 1.55],
       [3.85, 4.8 , 2.6 ],
       [1.15, 0.85, 4.15]])).numpy(), np.array([[13.7075, 19.5025, 13.26  ],
       [22.31  , 29.785 , 18.7275],
       [ 4.36  ,  4.365 , 10.22  ]]))
    np.testing.assert_allclose(ndl.matmul(ndl.Tensor([[3.8 , 0.05],
       [2.3 , 3.35],
       [1.6 , 2.6 ]]), ndl.Tensor([[1.1 , 3.5 , 3.7 ],
       [0.05, 1.25, 1.  ]])).numpy(), np.array([[ 4.1825, 13.3625, 14.11  ],
       [ 2.6975, 12.2375, 11.86  ],
       [ 1.89  ,  8.85  ,  8.52  ]]))
    np.testing.assert_allclose(ndl.matmul(ndl.Tensor([[[4.  , 2.15],
        [1.25, 1.35],
        [0.75, 1.6 ]],
       [[2.9 , 2.15],
        [3.3 , 4.1 ],
        [2.5 , 0.25]],
       [[2.9 , 4.35],
        [1.2 , 3.5 ],
        [3.55, 3.95]],
       [[2.55, 4.35],
        [4.25, 0.2 ],
        [3.95, 3.4 ]],
       [[2.2 , 2.05],
        [0.95, 1.8 ],
        [2.7 , 2.  ]],
       [[0.45, 1.1 ],
        [3.15, 0.7 ],
        [2.9 , 1.95]]]), ndl.Tensor([[[2.7 , 4.05, 0.1 ],
        [1.75, 3.05, 2.3 ]],
       [[0.55, 4.1 , 2.3 ],
        [4.45, 2.35, 2.55]],
       [[1.2 , 3.95, 4.6 ],
        [4.2 , 3.5 , 3.35]],
       [[2.55, 4.4 , 2.05],
        [2.4 , 0.6 , 4.65]],
       [[2.95, 0.8 , 0.6 ],
        [0.45, 1.3 , 0.75]],
       [[1.25, 2.1 , 0.4 ],
        [0.85, 3.5 , 3.7 ]]])).numpy(), np.array([[[14.5625, 22.7575,  5.345 ],
        [ 5.7375,  9.18  ,  3.23  ],
        [ 4.825 ,  7.9175,  3.755 ]],
       [[11.1625, 16.9425, 12.1525],
        [20.06  , 23.165 , 18.045 ],
        [ 2.4875, 10.8375,  6.3875]],
       [[21.75  , 26.68  , 27.9125],
        [16.14  , 16.99  , 17.245 ],
        [20.85  , 27.8475, 29.5625]],
       [[16.9425, 13.83  , 25.455 ],
        [11.3175, 18.82  ,  9.6425],
        [18.2325, 19.42  , 23.9075]],
       [[ 7.4125,  4.425 ,  2.8575],
        [ 3.6125,  3.1   ,  1.92  ],
        [ 8.865 ,  4.76  ,  3.12  ]],
       [[ 1.4975,  4.795 ,  4.25  ],
        [ 4.5325,  9.065 ,  3.85  ],
        [ 5.2825, 12.915 ,  8.375 ]]]))
    np.testing.assert_allclose(ndl.matmul(ndl.Tensor([[1.9 , 1.9 ],
       [4.8 , 4.9 ],
       [3.25, 3.75]]), ndl.Tensor([[[1.25, 1.8 , 1.95],
        [3.75, 2.85, 2.25]],
       [[1.75, 2.7 , 3.3 ],
        [2.95, 1.55, 3.85]],
       [[4.2 , 3.05, 3.35],
        [3.3 , 4.75, 2.1 ]]])).numpy(), np.array([[[ 9.5   ,  8.835 ,  7.98  ],
        [24.375 , 22.605 , 20.385 ],
        [18.125 , 16.5375, 14.775 ]],
       [[ 8.93  ,  8.075 , 13.585 ],
        [22.855 , 20.555 , 34.705 ],
        [16.75  , 14.5875, 25.1625]],
       [[14.25  , 14.82  , 10.355 ],
        [36.33  , 37.915 , 26.37  ],
        [26.025 , 27.725 , 18.7625]]]))
    np.testing.assert_allclose(ndl.matmul(ndl.Tensor([[[3.4 , 2.95],
        [0.25, 1.95],
        [4.4 , 4.4 ]],
       [[0.55, 1.1 ],
        [0.75, 1.55],
        [4.1 , 1.2 ]],
       [[1.5 , 4.05],
        [1.5 , 1.55],
        [2.3 , 1.25]]]), ndl.Tensor([[2.2 , 0.65, 2.5 ],
       [2.5 , 1.3 , 0.15]])).numpy(), np.array([[[14.855 ,  6.045 ,  8.9425],
        [ 5.425 ,  2.6975,  0.9175],
        [20.68  ,  8.58  , 11.66  ]],
       [[ 3.96  ,  1.7875,  1.54  ],
        [ 5.525 ,  2.5025,  2.1075],
        [12.02  ,  4.225 , 10.43  ]],
       [[13.425 ,  6.24  ,  4.3575],
        [ 7.175 ,  2.99  ,  3.9825],
        [ 8.185 ,  3.12  ,  5.9375]]]))


def test_summation_forward():
    np.testing.assert_allclose(ndl.summation(ndl.Tensor([[2.2 , 4.35, 1.4 , 0.3 , 2.65],
       [1.  , 0.85, 2.75, 3.8 , 1.55],
       [3.2 , 2.3 , 3.45, 0.7 , 0.  ]])).numpy(), np.array(30.5))
    np.testing.assert_allclose(ndl.summation(ndl.Tensor([[1.05, 2.55, 1.  ],
       [2.95, 3.7 , 2.6 ],
       [0.1 , 4.1 , 3.3 ],
       [1.1 , 3.4 , 3.4 ],
       [1.8 , 4.55, 2.3 ]]), axes=1).numpy(), np.array([4.6 , 9.25, 7.5 , 7.9 , 8.65]))
    np.testing.assert_allclose(ndl.summation(ndl.Tensor([[1.5 , 3.85, 3.45],
       [1.35, 1.3 , 0.65],
       [2.6 , 4.55, 0.25]]), axes=0).numpy(), np.array([5.45, 9.7 , 4.35]))


def test_broadcast_to_forward():
    np.testing.assert_allclose(ndl.broadcast_to(ndl.Tensor([[1.85, 0.85, 0.6 ]]), shape=(3, 3, 3)).numpy(), np.array([[[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]],
       [[1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ],
        [1.85, 0.85, 0.6 ]]]))


def test_reshape_forward():
    np.testing.assert_allclose(ndl.reshape(ndl.Tensor([[2.9 , 2.  , 2.4 ],
       [3.95, 3.95, 4.65],
       [2.1 , 2.5 , 2.7 ],
       [1.9 , 4.85, 3.25],
       [3.35, 3.45, 3.45]]), shape=(15,)).numpy(), np.array([2.9 , 2.  , 2.4 , 3.95, 3.95, 4.65, 2.1 , 2.5 , 2.7 , 1.9 , 4.85,
       3.25, 3.35, 3.45, 3.45]))
    np.testing.assert_allclose(ndl.reshape(ndl.Tensor([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15]],
       [[4.7 , 1.4 , 2.55, 4.8 ],
        [2.8 , 1.75, 2.8 , 0.6 ]],
       [[3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]), shape=(2, 3, 4)).numpy(), np.array([[[4.1 , 4.05, 1.35, 1.65],
        [3.65, 0.9 , 0.65, 4.15],
        [4.7 , 1.4 , 2.55, 4.8 ]],
       [[2.8 , 1.75, 2.8 , 0.6 ],
        [3.75, 0.6 , 0.  , 3.5 ],
        [0.15, 1.9 , 4.75, 2.8 ]]]))

def test_negate_forward():
    np.testing.assert_allclose(ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy(), np.array([[-1.45, -0.55]]))


def test_transpose_forward():
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[[1.95]],
       [[2.7 ]],
       [[3.75]]]), axes=(1, 2)).numpy(), np.array([[[1.95]],
       [[2.7 ]],
       [[3.75]]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]), axes=(2, 3)).numpy(), np.array([[[[0.95]]],
       [[[2.55]]],
       [[[0.45]]]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[[[0.4 , 0.05],
         [2.95, 1.3 ]],
        [[4.8 , 1.2 ],
         [1.65, 3.1 ]]],
       [[[1.45, 3.05],
         [2.25, 0.1 ]],
        [[0.45, 4.75],
         [1.5 , 1.8 ]]],
       [[[1.5 , 4.65],
         [1.35, 2.7 ]],
        [[2.  , 1.65],
         [2.05, 1.2 ]]]])).numpy(), np.array([[[[0.4 , 2.95],
         [0.05, 1.3 ]],
        [[4.8 , 1.65],
         [1.2 , 3.1 ]]],
       [[[1.45, 2.25],
         [3.05, 0.1 ]],
        [[0.45, 1.5 ],
         [4.75, 1.8 ]]],
       [[[1.5 , 1.35],
         [4.65, 2.7 ]],
        [[2.  , 2.05],
         [1.65, 1.2 ]]]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[[2.45]],
       [[3.5 ]],
       [[0.9 ]]]), axes=(0, 1)).numpy(), np.array([[[2.45],
        [3.5 ],
        [0.9 ]]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[4.4 , 2.05],
       [1.85, 2.25],
       [0.15, 1.4 ]])).numpy(), np.array([[4.4 , 1.85, 0.15],
       [2.05, 2.25, 1.4 ]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[0.05, 3.7 , 1.35],
       [4.45, 3.25, 1.95],
       [2.45, 4.4 , 4.5 ]])).numpy(), np.array([[0.05, 4.45, 2.45],
       [3.7 , 3.25, 4.4 ],
       [1.35, 1.95, 4.5 ]]))
    np.testing.assert_allclose(ndl.transpose(ndl.Tensor([[[0.55, 1.8 , 0.2 ],
        [0.8 , 2.75, 3.7 ],
        [0.95, 1.4 , 0.8 ]],
       [[0.75, 1.6 , 1.35],
        [3.75, 4.  , 4.55],
        [1.85, 2.5 , 4.8 ]],
       [[0.2 , 3.35, 3.4 ],
        [0.3 , 4.85, 4.85],
        [4.35, 4.25, 3.05]]]), axes=(0, 1)).numpy(), np.array([[[0.55, 1.8 , 0.2 ],
        [0.75, 1.6 , 1.35],
        [0.2 , 3.35, 3.4 ]],
       [[0.8 , 2.75, 3.7 ],
        [3.75, 4.  , 4.55],
        [0.3 , 4.85, 4.85]],
       [[0.95, 1.4 , 0.8 ],
        [1.85, 2.5 , 4.8 ],
        [4.35, 4.25, 3.05]]]))


def submit_forward():
    mugrade.submit(ndl.divide(ndl.Tensor([[3.4 , 2.35, 1.25 ], [0.45, 1.95, 2.55]]),
                              ndl.Tensor([[4.9 , 4.35, 4.1 ], [0.65, 0.7 , 4.04 ]])).numpy())
    mugrade.submit(ndl.divide_scalar(ndl.Tensor([[1.4 , 2.89]]), scalar=7).numpy())
    mugrade.submit(ndl.matmul(ndl.Tensor([[1.75, 1.75, 0.25], [4.95, 4.35, 0.3], [0.3, 1.4, 2.1]]),
                              ndl.Tensor([[2.35, 2.2, 1.85], [7.85, 4.88, 2.6], [1.15, 0.25, 4.19]])).numpy())
    mugrade.submit(ndl.summation(ndl.Tensor([[1.2, 4.35, 1.4, 0.3, 0.75],
                                             [2., 1.85, 7.75, 3.7, 1.55],
                                             [9.2, 2.3, 3.45, 0.7, 0.]])).numpy())
    mugrade.submit(ndl.summation(ndl.Tensor([[5.05, 2.55, 1.],
                                             [2.75, 3.7, 2.1],
                                             [0.1, 4.1, 3.3],
                                             [1.4, 0.4, 3.4],
                                             [2.8, 0.55, 2.9]]), axes=1).numpy())
    mugrade.submit(ndl.broadcast_to(ndl.Tensor([[1.95, 3.85, -0.6]]), shape=(3, 3, 3)).numpy())
    mugrade.submit(ndl.reshape(ndl.Tensor([[7.9, 2., 2.4],
                                           [3.11, 3.95, 0.65],
                                           [2.1, 2.18, 2.2],
                                           [1.9, 4.54, 3.25],
                                           [1.35, 7.45, 3.45]]), shape=(15,)).numpy())
    mugrade.submit(ndl.reshape(ndl.Tensor([[[5.1, 4.05, 1.25, 4.65],
                                            [3.65, 0.9, 0.65, 1.65]],
                                           [[4.7, 1.4, 2.55, 4.8],
                                            [2.8, 1.75, 3.8, 0.6]],
                                           [[3.75, 0.6, 1., 3.5],
                                            [8.15, 1.9, 4.55, 2.83]]]), shape=(2, 3, 4)).numpy())
    mugrade.submit(ndl.negate(ndl.Tensor([[1.45, 0.55]])).numpy())
    mugrade.submit(ndl.transpose(ndl.Tensor([[[3.45]],
                                             [[2.54]],
                                             [[1.91]]]), axes=(0, 1)).numpy())
    mugrade.submit(ndl.transpose(ndl.Tensor([[4.45, 2.15],
                                             [1.89, 1.21],
                                             [6.15, 2.42]])).numpy())

##############################################################################
### TESTS/SUBMISSION CODE FOR backward passes

def gradient_check(f, *args, tol=1e-6, backward=False, **kwargs):
    eps = 1e-4
    numerical_grads = [np.zeros(a.shape) for a in args]
    for i in range(len(args)):
        for j in range(args[i].realize_cached_data().size):
            args[i].realize_cached_data().flat[j] += eps
            f1 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] -= 2 * eps
            f2 = float(f(*args, **kwargs).numpy().sum())
            args[i].realize_cached_data().flat[j] += eps
            numerical_grads[i].flat[j] = (f1 - f2) / (2 * eps)
    if not backward:
        out = f(*args, **kwargs)
        computed_grads = [x.numpy() for x in out.op.gradient_as_tuple(ndl.Tensor(np.ones(out.shape)), out)]
    else:
        out = f(*args, **kwargs).sum()
        out.backward()
        computed_grads = [a.grad.numpy() for a in args]
    error = sum(
        np.linalg.norm(computed_grads[i] - numerical_grads[i])
        for i in range(len(args))
    )
    assert error < tol
    return computed_grads


def test_divide_backward():
    gradient_check(ndl.divide, ndl.Tensor(np.random.randn(5, 4)), ndl.Tensor(5 + np.random.randn(5, 4)))


def test_divide_scalar_backward():
    gradient_check(ndl.divide_scalar, ndl.Tensor(np.random.randn(5, 4)), scalar=np.random.randn(1))


def test_matmul_simple_backward():
    gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(5, 4)), ndl.Tensor(np.random.randn(4, 5)))


def test_matmul_batched_backward():
    gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(6, 6, 5, 4)), ndl.Tensor(np.random.randn(6, 6, 4, 3)))
    gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(6, 6, 5, 4)), ndl.Tensor(np.random.randn(4, 3)))
    gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(5, 4)), ndl.Tensor(np.random.randn(6, 6, 4, 3)))


def test_reshape_backward():
    gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(4, 5))


def test_negate_backward():
    gradient_check(ndl.negate, ndl.Tensor(np.random.randn(5, 4)))


def test_transpose_backward():
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(1, 2))
    gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 5, 4)), axes=(0, 1))

def test_broadcast_to_backward():
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(3, 1)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 3)), shape=(3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1,)), shape=(3, 3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(3, 3, 3))
    gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(5,4,1)), shape=(5,4,3))


def test_summation_backward():
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(1,))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(0,))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4)), axes=(0,1))
    gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4,1)), axes=(0,1))


def submit_backward():
    np.random.seed(0)
    out = gradient_check(ndl.divide, ndl.Tensor(np.random.randn(3, 5)), ndl.Tensor(6 + np.random.randn(3, 5)))
    print(out)
    mugrade.submit(out)
    mugrade.submit(gradient_check(ndl.divide_scalar, ndl.Tensor(np.random.randn(3, 5)), scalar=np.random.randn(1)))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(1, 5)), ndl.Tensor(np.random.randn(5, 1))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(4, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(7, 4, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(3, 2, 1)), ndl.Tensor(np.random.randn(3, 3, 1, 2))))
    mugrade.submit(gradient_check(ndl.matmul, ndl.Tensor(np.random.randn(2, 4)), ndl.Tensor(np.random.randn(2, 4, 4, 2))))
    mugrade.submit(gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(5,4,1)))
    mugrade.submit(gradient_check(ndl.reshape, ndl.Tensor(np.random.randn(5, 4)), shape=(2, 2, 5)))
    mugrade.submit(gradient_check(ndl.negate, ndl.Tensor(np.random.randn(1, 4, 2))))
    mugrade.submit(gradient_check(ndl.transpose, ndl.Tensor(np.random.randn(3, 2, 4)), axes=(0, 2)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(7, 1)), shape=(7, 7)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1, 5)), shape=(5, 5)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(1,)), shape=(4, 4, 4)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn()), shape=(1, 3, 6)))
    mugrade.submit(gradient_check(ndl.broadcast_to, ndl.Tensor(np.random.randn(4,4,1)), shape=(4,4,6)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3,2,1))))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(3,6)), axes=(1,)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(7,)), axes=(0,)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(7,8)), axes=(0,1)))
    mugrade.submit(gradient_check(ndl.summation, ndl.Tensor(np.random.randn(5,4,5)), axes=(0,1,2)))


##############################################################################
### TESTS/SUBMISSION CODE FOR find_topo_sort

def test_topo_sort():
    # Test case 1
    a1, b1 = ndl.Tensor(np.asarray([[0.88282157]])), ndl.Tensor(np.asarray([[0.90170084]]))
    c1 = 3*a1*a1 + 4*b1*a1 - a1

    soln = np.array([np.array([[0.88282157]]),
                     np.array([[2.64846471]]),
                     np.array([[2.33812177]]),
                     np.array([[0.90170084]]),
                     np.array([[3.60680336]]),
                     np.array([[3.1841638]]),
                     np.array([[5.52228558]]),
                     np.array([[-0.88282157]]),
                     np.array([[4.63946401]])])

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c1])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)

    # Test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.20914675], [0.65264178]])), ndl.Tensor(np.asarray([[0.65394286, 0.08218317]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    soln = [np.array([[0.65394286, 0.08218317]]),
            np.array([[0.20914675], [0.65264178]]),
            np.array([[0.19040619]]),
            np.array([[1.53101102, 0.19240724]]),
            np.array([[0.44577898]]), np.array([[0.63618518]]),
            np.array([[1.90855553]]), np.array([[3.40855553]])]

    topo_order = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    assert len(soln) == len(topo_order)
    # step through list as entries differ in length
    for t, s in zip(topo_order, soln):
        np.testing.assert_allclose(t, s, rtol=1e-06, atol=1e-06)

    # Test case 3
    a = ndl.Tensor(np.asarray([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]))
    b = ndl.Tensor(np.asarray([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]))
    e = (a@b + b - a)@a

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([e])])

    soln = np.array([np.array([[1.4335016, 0.30559972], [0.08130171, -1.15072371]]),
                     np.array([[1.34571691, -0.95584433], [-0.99428573, -0.04017499]]),
                     np.array([[1.6252339, -1.38248184], [1.25355725, -0.03148146]]),
                     np.array([[2.97095081, -2.33832617], [0.25927152, -0.07165645]]),
                     np.array([[-1.4335016, -0.30559972], [-0.08130171, 1.15072371]]),
                     np.array([[1.53744921, -2.64392589], [0.17796981, 1.07906726]]),
                     np.array([[1.98898021, 3.51227226], [0.34285002, -1.18732075]])])

    assert len(soln) == len(topo_order)
    np.testing.assert_allclose(topo_order, soln, rtol=1e-06, atol=1e-06)


def submit_topo_sort():

    # mugrade test case 1
    a2, b2 = ndl.Tensor(np.asarray([[0.74683138]])), ndl.Tensor(np.asarray([[0.65539231]]))
    c2 = 9 * a2 * a2 + 15 * b2 * a2 - b2

    topo_order = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([c2])])

    mugrade.submit(topo_order)


    # mugrade test case 2
    a1, b1 = ndl.Tensor(np.asarray([[0.9067453], [0.18521121]])), ndl.Tensor(np.asarray([[0.80992494, 0.52458167]]))
    c1 = 3 * ((b1 @ a1) + (2.3412 * b1) @ a1) + 1.5

    topo_order2 = [x.numpy() for x in ndl.autograd.find_topo_sort([c1])]

    mugrade.submit(topo_order2)

    # mugrade test case 3
    c = ndl.Tensor(np.asarray([[-0.16541387, 2.52604789], [-0.31008569, -0.4748876]]))
    d = ndl.Tensor(np.asarray([[0.55936155, -2.12630983], [0.59930618, -0.19554253]]))
    f = (c + d@d - d) @ c

    topo_order3 = np.array([x.numpy() for x in ndl.autograd.find_topo_sort([f])])

    mugrade.submit(topo_order3)


##############################################################################
### TESTS/SUBMISSION CODE FOR compute_gradient_of_variables

def test_compute_gradient():
    gradient_check(lambda A,B,C : ndl.summation((A@B+C)*(A@B), axes=None),
                   ndl.Tensor(np.random.randn(10,9)),
                   ndl.Tensor(np.random.randn(9,8)),
                   ndl.Tensor(np.random.randn(10,8)), backward=True)
    gradient_check(lambda A,B : ndl.summation(ndl.broadcast_to(A,shape=(10,9))*B, axes=None),
                   ndl.Tensor(np.random.randn(10,1)),
                   ndl.Tensor(np.random.randn(10,9)), backward=True)
    gradient_check(lambda A,B,C : ndl.summation(ndl.reshape(A,shape=(10,10))@B/5+C, axes=None),
                   ndl.Tensor(np.random.randn(100)),
                   ndl.Tensor(np.random.randn(10,5)),
                   ndl.Tensor(np.random.randn(10,5)), backward=True)

    # check gradient of gradient
    x2 = ndl.Tensor([6])
    x3 = ndl.Tensor([0])
    y = x2 * x2 + x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    assert y.numpy() == x2_val * x2_val + x2_val * x3_val
    assert grad_x2.numpy() == 2 * x2_val + x3_val
    assert grad_x3.numpy() == x2_val
    assert grad_x2_x2.numpy() == 2
    assert grad_x2_x3.numpy() == 1


def submit_compute_gradient():
    a = ndl.Tensor(np.array([[-0.2985143, 0.36875625],
                             [-0.918687, 0.52262925]]))
    b = ndl.Tensor(np.array([[-1.58839928, 1.58592338],
                             [-0.15932137, -0.55618462]]))
    c = ndl.Tensor(np.array([[-0.5096208, 0.73466865],
                             [0.38762148, -0.41149092]]))
    d = (a + b)@c@(a + c)
    d.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    mugrade.submit(grads)

    # just need a fixed function or two to send results to mugrade
    a = ndl.Tensor(np.array([[0.4736625, 0.06895066, 1.36455087, -0.31170743, 0.1370395],
                             [0.2283258, 0.72298311, -1.20394586, -1.95844434, -0.69535299],
                             [0.18016408, 0.0266557, 0.80940201, -0.45913678, -0.05886218],
                             [-0.50678819, -1.53276348, -0.27915708, -0.571393, -0.17145921]]))
    b = ndl.Tensor(np.array([[0.28738358, -1.27265428, 0.32388374],
                             [-0.77830395, 2.07830592, 0.99796268],
                             [-0.76966429, -1.37012833, -0.16733693],
                             [-0.44134101, -1.24495901, -1.62953897],
                             [-0.75627713, -0.80006226, 0.03875171]]))
    c = ndl.Tensor(np.array([[1.25727301, 0.39400789, 1.29139323, -0.950472],
                             [-0.21250305, -0.93591609, 1.6802009, -0.39765765],
                             [-0.16926597, -0.45218718, 0.38103032, -0.11321965]]))
    output = ndl.summation((a@b)@c@a)
    output.backward()
    grads = [x.grad.numpy() for x in [a, b, c]]
    mugrade.submit(grads)

    x2 = ndl.Tensor(3)
    x3 = ndl.Tensor(2)
    y = x2 * x2 - x2 * x3
    y.backward()
    grad_x2 = x2.grad
    grad_x3 = x3.grad
    # gradient of gradient
    grad_x2.backward()
    grad_x2_x2 = x2.grad
    grad_x2_x3 = x3.grad
    x2_val = x2.numpy()
    x3_val = x3.numpy()
    mugrade.submit(y.numpy())
    mugrade.submit(grad_x2.numpy())
    mugrade.submit(grad_x3.numpy())
    mugrade.submit(grad_x2_x2.numpy())

##############################################################################
### TESTS/SUBMISSION CODE FOR softmax_loss

def test_softmax_loss_ndl():
    # test forward pass for log
    np.testing.assert_allclose(ndl.log(ndl.Tensor([[4.  ],
       [4.55]])).numpy(), np.array([[1.38629436112 ],
       [1.515127232963]]))

    # test backward pass for log
    gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5,4)))

    # test backward pass for exp
    gradient_check(ndl.exp, ndl.Tensor(np.random.rand(5, 4)))

    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    Z = ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32))
    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    np.testing.assert_allclose(softmax_loss(Z,y).numpy(), 2.3025850, rtol=1e-6, atol=1e-6)
    Z = ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32))
    np.testing.assert_allclose(softmax_loss(Z,y).numpy(), 2.7291998, rtol=1e-6, atol=1e-6)

    # test softmax loss backward
    Zsmall = ndl.Tensor(np.random.randn(16, 10).astype(np.float32))
    ysmall = ndl.Tensor(y_one_hot[:16])
    gradient_check(softmax_loss, Zsmall, ysmall, tol=0.01, backward=True)


def submit_softmax_loss_ndl():
    # add a mugrade submit for log
    np.random.seed(0)
    mugrade.submit(gradient_check(ndl.log, ndl.Tensor(1 + np.random.rand(5, 4))))

    X,y = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                      "data/t10k-labels-idx1-ubyte.gz")

    y_one_hot = np.zeros((y.shape[0], 10))
    y_one_hot[np.arange(y.size), y] = 1
    y = ndl.Tensor(y_one_hot)
    mugrade.submit(softmax_loss(ndl.Tensor(np.zeros((y.shape[0], 10)).astype(np.float32)), y).numpy())
    np.random.seed(0)
    mugrade.submit(softmax_loss(ndl.Tensor(np.random.randn(y.shape[0], 10).astype(np.float32)), y).numpy())


##############################################################################
### TESTS/SUBMISSION CODE FOR nn_epoch

def test_nn_epoch_ndl():
    # test forward/backward pass for relu
    np.testing.assert_allclose(ndl.relu(ndl.Tensor([[-46.9 , -48.8 , -45.45, -49.  ],
       [-49.75, -48.75, -45.8 , -49.25],
       [-45.65, -45.25, -49.3 , -47.65]])).numpy(), np.array([[0., 0., 0., 0.],
       [0., 0., 0., 0.],
       [0., 0., 0., 0.]]))
    gradient_check(ndl.relu, ndl.Tensor(np.random.randn(5,4)))

    # test nn gradients
    np.random.seed(0)
    X = np.random.randn(50,5).astype(np.float32)
    y = np.random.randint(3, size=(50,)).astype(np.uint8)
    W1 = np.random.randn(5, 10).astype(np.float32) / np.sqrt(10)
    W2 = np.random.randn(10, 3).astype(np.float32) / np.sqrt(3)
    W1_0, W2_0 = W1.copy(), W2.copy()
    W1 = ndl.Tensor(W1)
    W2 = ndl.Tensor(W2)
    X_ = ndl.Tensor(X)
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    dW1 = nd.Gradient(lambda W1_ :
        softmax_loss(ndl.relu(X_@ndl.Tensor(W1_).reshape((5,10)))@W2, y_).numpy())(W1.numpy())
    dW2 = nd.Gradient(lambda W2_ :
        softmax_loss(ndl.relu(X_@W1)@ndl.Tensor(W2_).reshape((10,3)), y_).numpy())(W2.numpy())
    W1, W2 = nn_epoch(X, y, W1, W2, lr=1.0, batch=50)
    np.testing.assert_allclose(dW1.reshape(5,10), W1_0-W1.numpy(), rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(dW2.reshape(10,3), W2_0-W2.numpy(), rtol=1e-4, atol=1e-4)

    # test full epoch
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(0)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)
    np.testing.assert_allclose(np.linalg.norm(W1.numpy()), 28.437788,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(W2.numpy()), 10.455095,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(loss_err(ndl.relu(ndl.Tensor(X)@W1)@W2, y),
                               (0.19770025, 0.06006667), rtol=1e-4, atol=1e-4)


def submit_nn_epoch_ndl():
    X,y = parse_mnist("data/train-images-idx3-ubyte.gz",
                      "data/train-labels-idx1-ubyte.gz")
    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X[:100], y[:100], W1, W2, lr=0.1, batch=100)

    mugrade.submit(np.linalg.norm(W1.numpy()))
    mugrade.submit(np.linalg.norm(W2.numpy()))

    np.random.seed(1)
    W1 = ndl.Tensor(np.random.randn(X.shape[1], 100).astype(np.float32) / np.sqrt(100))
    W2 = ndl.Tensor(np.random.randn(100, 10).astype(np.float32) / np.sqrt(10))
    W1, W2 = nn_epoch(X, y, W1, W2, lr=0.2, batch=100)

    mugrade.submit(np.linalg.norm(W1.numpy()))
    mugrade.submit(np.linalg.norm(W2.numpy()))
    mugrade.submit(loss_err(ndl.Tensor(np.maximum(X@W1.numpy(),0))@W2, y))
