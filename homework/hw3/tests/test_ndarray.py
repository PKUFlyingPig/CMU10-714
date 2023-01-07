import sys
sys.path.append("./python")
import numpy as np
import pytest
import mugrade
import needle as ndl
from needle import backend_ndarray as nd


_DEVICES = [nd.cpu(), pytest.param(nd.cuda(), 
    marks=pytest.mark.skipif(not nd.cuda().enabled(), reason="No GPU"))]


def compare_strides(a_np, a_nd):
    size = a_np.itemsize
    assert tuple([x // size for x in a_np.strides]) == a_nd.strides


def check_same_memory(original, view):
    assert original._handle.ptr() == view._handle.ptr()


# TODO test permute, broadcast_to, reshape, getitem, some combinations thereof
@pytest.mark.parametrize("params", [
    {
     "shape": (4, 4),
     "np_fn": lambda X: X.transpose(),
     "nd_fn": lambda X: X.permute((1, 0))
    },
    {
     "shape": (4, 1, 4),
     "np_fn": lambda X: np.broadcast_to(X, shape=(4, 5, 4)),
     "nd_fn": lambda X: X.broadcast_to((4, 5, 4))
    },
    {
     "shape": (4, 3),
     "np_fn": lambda X: X.reshape(2, 2, 3),
     "nd_fn": lambda X: X.reshape((2, 2, 3))
    },
    {
     "shape": (16, 16), # testing for compaction of large ndims array
     "np_fn": lambda X: X.reshape(2, 4, 2, 2, 2, 2, 2),
     "nd_fn": lambda X: X.reshape((2, 4, 2, 2, 2, 2, 2))
    },
    {
     "shape": (2, 4, 2, 2, 2, 2, 2), # testing for compaction of large ndims array
     "np_fn": lambda X: X.reshape(16, 16),
     "nd_fn": lambda X: X.reshape((16, 16))
    },
    {
     "shape": (8, 8),
     "np_fn": lambda X: X[4:,4:],
     "nd_fn": lambda X: X[4:,4:]
    },
    {
     "shape": (8, 8, 2, 2, 2, 2),
     "np_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2],
     "nd_fn": lambda X: X[1:3, 5:8, 1:2, 0:1, 0:1, 1:2]
    }, 
    {
     "shape": (7, 8),
     "np_fn": lambda X: X.transpose()[3:7,2:5],
     "nd_fn": lambda X: X.permute((1, 0))[3:7,2:5]
    },   
], ids=["transpose", "broadcast_to", "reshape1", "reshape2", "reshape3", "getitem1", "getitem2", "transposegetitem"])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_compact(params, device):
    shape, np_fn, nd_fn = params['shape'], params['np_fn'], params['nd_fn']
    _A = np.random.randint(low=0, high=10, size=shape)
    A = nd.array(_A, device=device)
    
    lhs = nd_fn(A).compact()
    assert lhs.is_compact(), "array is not compact"

    rhs = np_fn(_A)
    np.testing.assert_allclose(lhs.numpy(), rhs, atol=1e-5, rtol=1e-5)


reduce_params = [
    {"dims": (10,), "axis": 0},
    {"dims": (4, 5, 6), "axis": 0},
    {"dims": (4, 5, 6), "axis": 1},
    {"dims": (4, 5, 6), "axis": 2}
]
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_sum(params, device):
    dims, axis = params['dims'], params['axis']
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)   
    np.testing.assert_allclose(_A.sum(axis=axis, keepdims=True), A.sum(axis=axis).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("params", reduce_params)
def test_reduce_max(params, device):
    dims, axis = params['dims'], params['axis']
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)   
    np.testing.assert_allclose(_A.max(axis=axis, keepdims=True), A.max(axis=axis).numpy(), atol=1e-5, rtol=1e-5)


""" For converting slice notation to slice objects to make some proceeding tests easier to read """
class _ShapeAndSlices(nd.NDArray):
    def __getitem__(self, idxs):
        idxs = tuple([self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)])
        return self.shape, idxs


ShapeAndSlices = lambda *shape: _ShapeAndSlices(np.ones(shape))


@pytest.mark.parametrize("params", [
    {
        "lhs": ShapeAndSlices(4, 5, 6)[1:2, 0, 0],
        "rhs": ShapeAndSlices(7, 7, 7)[1:2, 0, 0]
    },
    {
        "lhs": ShapeAndSlices(4, 5, 6)[1:4:2, 0, 0],
        "rhs": ShapeAndSlices(7, 7, 7)[1:3, 0, 0]
    },
    {
        "lhs": ShapeAndSlices(4, 5, 6)[1:3, 2:5, 2:6],
        "rhs": ShapeAndSlices(7, 7, 7)[:2, :3, :4]
    },   
])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_setitem_ewise(params, device):
    lhs_shape, lhs_slices = params['lhs']
    rhs_shape, rhs_slices = params['rhs']
    _A = np.random.randn(*lhs_shape)
    _B = np.random.randn(*rhs_shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    start_ptr = A._handle.ptr()
    A[lhs_slices] = B[rhs_slices]
    _A[lhs_slices] = _B[rhs_slices]
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    compare_strides(_A, A)
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)


# Ex: We want arrays of size (4, 5, 6) setting element(s) [1:4, 2, 3] to a scalar
@pytest.mark.parametrize("params", [
    ShapeAndSlices(4, 5, 6)[1,    2,   3],
    ShapeAndSlices(4, 5, 6)[1:4,  2,   3],
    ShapeAndSlices(4, 5, 6)[:4,  2:5, 3],
    ShapeAndSlices(4, 5, 6)[1::2, 2:5, ::2],
])
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"]) 
def test_setitem_scalar(params, device):
    shape, slices = params
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    # probably tear these out using lambdas
    print(slices)
    start_ptr = A._handle.ptr()
    _A[slices] = 4.0
    A[slices] = 4.0
    end_ptr = A._handle.ptr()
    assert start_ptr == end_ptr, "you should modify in-place"
    np.testing.assert_allclose(A.numpy(), _A, atol=1e-5, rtol=1e-5)
    compare_strides(_A, A)


matmul_tiled_shapes = [(1, 1, 1), (2, 2, 3), (1, 2, 1), (3, 3, 3)]
@pytest.mark.parametrize("m,n,p", matmul_tiled_shapes)
def test_matmul_tiled(m, n, p):
    device = nd.cpu()
    assert hasattr(device, "matmul_tiled")
    t = device.__tile_size__
    A = nd.array(np.random.randn(m, n, t, t), device=nd.cpu())
    B = nd.array(np.random.randn(n, p, t, t), device=nd.cpu())
    C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
    device.matmul_tiled(A._handle, B._handle, C._handle, m*t, n*t, p*t)

    lhs = A.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, n*t) \
        @ B.numpy().transpose(0, 2, 1, 3).flatten().reshape(n*t, p*t)
    rhs = C.numpy().transpose(0, 2, 1, 3).flatten().reshape(m*t, p*t)

    np.testing.assert_allclose(lhs, rhs, atol=1e-5, rtol=1e-5)


OPS = {
    "multiply": lambda a, b: a * b,
    "divide": lambda a, b: a / b,
    "add": lambda a, b: a + b,
    "subtract": lambda a, b: a - b,
    "equal": lambda a, b: a == b,
    "greater_than": lambda a, b: a >= b
}
OP_FNS = [OPS[k] for k in OPS]
OP_NAMES = [k for k in OPS]

ewise_shapes = [(1, 1, 1), (4, 5, 6)]
@pytest.mark.parametrize("fn", OP_FNS, ids=OP_NAMES)
@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_fn(fn, shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(fn(_A, _B), fn(A, B).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("shape", ewise_shapes)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_max(shape, device):
    _A = np.random.randn(*shape)
    _B = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose(np.maximum(_A, _B), A.maximum(B).numpy(), atol=1e-5, rtol=1e-5)


permute_params = [
    {"dims": (4, 5, 6), "axes": (0, 1, 2)},
    {"dims": (4, 5, 6), "axes": (1, 0, 2)},
    {"dims": (4, 5, 6), "axes": (2, 1, 0)},
]
@pytest.mark.parametrize("params", permute_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_permute(device, params):
    dims = params['dims']
    axes = params['axes']
    _A = np.random.randn(*dims)
    A = nd.array(_A, device=device)
    lhs = np.transpose(_A, axes=axes)
    rhs = A.permute(axes)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


reshape_params = [
    {"shape": (8, 16), "new_shape": (2, 4, 16)},
    {"shape": (8, 16), "new_shape": (8, 4, 2, 2)},
]
@pytest.mark.parametrize("params", reshape_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_reshape(device, params):
    shape = params['shape']
    new_shape = params['new_shape']
    _A = np.random.randn(*shape)
    A = nd.array(_A, device=device)
    lhs = _A.reshape(*new_shape)
    rhs = A.reshape(new_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


getitem_params = [
    {"shape": (8, 16), "fn": lambda X: X[3:4, 3:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:2, 1:3]},
    {"shape": (8, 16), "fn": lambda X: X[3:4, 1:4]},
    {"shape": (8, 16), "fn": lambda X: X[1:4:2, 3:4]},
]
@pytest.mark.parametrize("params", getitem_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_getitem(device, params):
    shape = params['shape']
    fn = params['fn']
    _A = np.random.randn(5, 5)
    A = nd.array(_A, device=device)
    lhs = fn(_A)
    rhs = fn(A)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


broadcast_params = [
    {"from_shape": (1, 3, 4), "to_shape": (6, 3, 4)},
]
@pytest.mark.parametrize("params", broadcast_params)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_broadcast_to(device, params):
    from_shape, to_shape = params['from_shape'], params['to_shape']
    _A = np.random.randn(*from_shape)
    A = nd.array(_A, device=device)
    lhs = np.broadcast_to(_A, shape=to_shape)
    rhs = A.broadcast_to(to_shape)
    np.testing.assert_allclose(lhs, rhs.numpy(), atol=1e-5, rtol=1e-5)
    compare_strides(lhs, rhs)
    check_same_memory(A, rhs)


matmul_dims = [(16, 16, 16), 
    (8, 8, 8), 
    (1, 2, 3), 
    (3, 4, 5), 
    (5, 4, 3), 
    (64, 64, 64), 
    (72, 72, 72), 
    (72, 73, 74), 
    (74, 73, 72), 
    (128, 128, 128)]
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
@pytest.mark.parametrize("m,n,p", matmul_dims)
def test_matmul(m, n, p, device):
    _A = np.random.randn(m, n)
    _B = np.random.randn(n, p)
    A = nd.array(_A, device=device)
    B = nd.array(_B, device=device)
    np.testing.assert_allclose((A @ B).numpy(), _A @ _B, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_mul(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(A * 5., (B * 5.).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_div(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(A / 5., (B / 5.).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_power(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.power(A, 5.), (B**5.).numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(np.power(A, 0.5), (B**0.5).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_maximum(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = (np.max(A) + 1.).item()
    np.testing.assert_allclose(np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5)
    C = (np.max(A) - 1.).item()
    np.testing.assert_allclose(np.maximum(A, C), (B.maximum(C)).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_eq(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0,1].item()
    np.testing.assert_allclose(A == C, (B == C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_scalar_ge(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    C = A[0, 1].item()
    np.testing.assert_allclose(A >= C, (B >= C).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_log(device):
    A = np.abs(np.random.randn(5, 5))
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.log(A), (B.log()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_exp(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.exp(A), (B.exp()).numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_ewise_tanh(device):
    A = np.random.randn(5, 5)
    B = nd.array(A, device=device)
    np.testing.assert_allclose(np.tanh(A), (B.tanh()).numpy(), atol=1e-5, rtol=1e-5)


######################    |    ######################
###################### MUGRADE ######################
######################    v    ######################

def Prepare(A):
    return (A.numpy().flatten()[:128], A.strides, A.shape)


def Rand(*shape, device=nd.cpu(), entropy=1):
    np.random.seed(np.prod(shape) * len(shape) * entropy)
    _A = np.random.randint(low=1, high=100, size=shape)
    return nd.array(_A, device=device)


def RandC(*shape, entropy=1):
    if nd.cuda().enabled():
        return Rand(*shape, device=nd.cuda(), entropy=2)
    else:
        raise NotImplementedError("You need a GPU to run these tests.")


def MugradeSubmit(things):
    mugrade.submit(Prepare(things))


def submit_ndarray_python_ops():
    MugradeSubmit(Rand(4, 4).reshape((2, 2, 4)))
    MugradeSubmit(Rand(2, 2, 4).reshape((4, 4)))
    MugradeSubmit(Rand(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)))
    MugradeSubmit(Rand(2, 1, 2).broadcast_to((2, 3, 2)))
    MugradeSubmit(Rand(1, 1, 2).broadcast_to((2, 2, 2)))
    MugradeSubmit(Rand(4, 4)[1, 0])
    MugradeSubmit(Rand(4, 4)[1, 0:3])
    MugradeSubmit(Rand(4, 4)[1:3, 0:3])
    MugradeSubmit(Rand(4, 4, 4)[0:2, 1:3, 2:4])
    MugradeSubmit(Rand(4, 4, 4)[0:4:2, :3, 2:])


def submit_ndarray_cpu_compact_setitem():
    MugradeSubmit(Rand(4, 4, 4)[0:4:2, :3, 2:].compact())
    MugradeSubmit(Rand(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)).compact())
    MugradeSubmit(Rand(1, 1, 2).broadcast_to((2, 2, 2)).compact())
    MugradeSubmit(Rand(4, 4).reshape((2, 2, 4)).compact())

    A = Rand(4, 4)
    B = Rand(4, 4)
    A[1, 0:3] = B[0, 1:4]
    MugradeSubmit(A)

    A = Rand(4, 4)
    B = Rand(4, 4)
    A[0:3, 1] = B[1:4, 0]
    MugradeSubmit(A)

    A = Rand(2, 2, 2, 3)
    B = Rand(2, 2, 2, 3)
    A[0, :, 1, :2] = B[0, :, 1, :2]
    MugradeSubmit(A)

    A = Rand(2, 2, 2, 3)
    A[0, :, 1, :2] = 42.0
    MugradeSubmit(A)


def submit_ndarray_cpu_ops():
    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A * B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A / B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A == B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A >= B)

    A, B = Rand(3, 3), Rand(3, 3)
    MugradeSubmit(A.maximum(B))

    A = Rand(2, 2)
    MugradeSubmit(A * 42.0)

    A = Rand(2, 2)
    MugradeSubmit(A ** 2.0)

    A = Rand(2, 2)
    MugradeSubmit(A / 42.0)

    A = Rand(5, 5)
    MugradeSubmit(A.maximum(25.0))

    A = Rand(10, 10)
    MugradeSubmit(A == 10)

    A = Rand(5, 5)
    MugradeSubmit(A > 50)

    A = Rand(2, 2)
    MugradeSubmit(A.log())

    A = Rand(2, 2)
    MugradeSubmit(A.tanh())

    A = Rand(2, 2)
    MugradeSubmit((A/100).exp())


def submit_ndarray_cpu_reductions():
    MugradeSubmit(Rand(4, 4).sum(axis=1))
    MugradeSubmit(Rand(4, 4, 4).sum(axis=1))
    MugradeSubmit(Rand(4).sum(axis=0))
    MugradeSubmit(Rand(2, 2, 2, 4, 2).sum(axis=3))
    MugradeSubmit(Rand(4, 4).max(axis=1))
    MugradeSubmit(Rand(4, 4, 4).max(axis=1))
    MugradeSubmit(Rand(4).max(axis=0))
    MugradeSubmit(Rand(2, 2, 2, 4, 2).max(axis=3))


def submit_ndarray_cpu_matmul():
    A, B = Rand(4, 4), Rand(4, 4)
    MugradeSubmit(A @ B)

    A, B = Rand(3, 4), Rand(4, 3)
    MugradeSubmit(A @ B)

    A, B = Rand(73, 72), Rand(72, 73)
    MugradeSubmit(A @ B)

    # tiled
    for m, n, p in [(3, 2, 1), (3, 3, 3), (3, 4, 5)]:
        device = nd.cpu()
        t = device.__tile_size__
        A = Rand(m, n, t, t)
        B = Rand(n, p, t, t)
        C = nd.NDArray.make((m, p, t, t), device=nd.cpu())
        device.matmul_tiled(A._handle, B._handle, C._handle, m*t, n*t, p*t)
        MugradeSubmit(C)


def submit_ndarray_cuda_compact_setitem():
    MugradeSubmit(RandC(4, 4, 4)[0:4:2, :3, 2:].compact())
    MugradeSubmit(RandC(1, 3, 2, 1, 2).permute((1, 3, 2, 0, 4)).compact())
    MugradeSubmit(RandC(1, 1, 2).broadcast_to((2, 2, 2)).compact())
    MugradeSubmit(RandC(4, 4).reshape((2, 2, 4)).compact())

    A = RandC(4, 4)
    B = RandC(4, 4)
    A[1, 0:3] = B[0, 1:4]
    MugradeSubmit(A)

    A = RandC(4, 4)
    B = RandC(4, 4)
    A[0:3, 1] = B[1:4, 0]
    MugradeSubmit(A)

    A = RandC(2, 2, 2, 3)
    B = RandC(2, 2, 2, 3)
    A[0, :, 1, :2] = B[0, :, 1, :2]
    MugradeSubmit(A)

    A = RandC(2, 2, 2, 3)
    A[0, :, 1, :2] = 42.0
    MugradeSubmit(A)


def submit_ndarray_cuda_ops():
    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A * B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A / B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A == B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A >= B)

    A, B = RandC(3, 3), RandC(3, 3)
    MugradeSubmit(A.maximum(B))

    A = RandC(2, 2)
    MugradeSubmit(A * 42.0)

    A = RandC(2, 2)
    MugradeSubmit(A ** 2.0)

    A = RandC(2, 2)
    MugradeSubmit(A / 42.0)

    A = RandC(5, 5)
    MugradeSubmit(A.maximum(25.0))

    A = RandC(10, 10)
    MugradeSubmit(A == 10)

    A = RandC(5, 5)
    MugradeSubmit(A > 50)

    A = RandC(2, 2)
    MugradeSubmit(A.log())

    A = RandC(2, 2)
    MugradeSubmit(A.tanh())

    A = RandC(2, 2)
    MugradeSubmit((A/100).exp())


def submit_ndarray_cuda_reductions():
    MugradeSubmit(RandC(4, 4).sum(axis=1))
    MugradeSubmit(RandC(4, 4, 4).sum(axis=1))
    MugradeSubmit(RandC(4).sum(axis=0))
    MugradeSubmit(RandC(2, 2, 2, 4, 2).sum(axis=3))
    MugradeSubmit(RandC(4, 4).max(axis=1))
    MugradeSubmit(RandC(4, 4, 4).max(axis=1))
    MugradeSubmit(RandC(4).max(axis=0))
    MugradeSubmit(RandC(2, 2, 2, 4, 2).max(axis=3))


def submit_ndarray_cuda_matmul():
    A, B = RandC(4, 4), RandC(4, 4)
    MugradeSubmit(A @ B)

    A, B = RandC(4, 3), RandC(3, 4)
    MugradeSubmit(A @ B)

    A, B = RandC(3, 4), RandC(4, 3)
    MugradeSubmit(A @ B)

    A, B = RandC(33, 33), RandC(33, 33)
    MugradeSubmit(A @ B)

    A, B = RandC(73, 72), RandC(72, 71)
    MugradeSubmit(A @ B)

    A, B = RandC(123, 125), RandC(125, 129)
    MugradeSubmit(A @ B)


if __name__ == "__main__":
    print("You have to run the tests with pytest due to parameterization.")