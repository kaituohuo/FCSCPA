import numpy as np

from FCSCPA import broyden_solver, np_inv_along_axis12

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def test_func0():
    hf1 = lambda x: x**2-1
    x0 = np.random.randn(1)
    x1 = broyden_solver(hf1, x0)
    tmp0 = np.abs(hf1(x1)).max()
    assert tmp0 < 1e-7


def test_func1():
    hf1 = lambda x: np.array([np.exp(-np.exp(x[1]-x[0])) - x[1]*(1+x[0]**2),
            x[0]*np.cos(x[1]) + x[1]*np.sin(x[0])-0.5])
    x0 = np.random.rand(2)
    x1 = broyden_solver(hf1, x0)
    tmp0 = np.abs(hf1(x1)).max()
    assert tmp0 < 1e-6

def test_func2():
    # still diverge sometimes
    x1_ = np.array([[1,2],[3,4]])
    y1 = np.linalg.matrix_power(x1_, 2).reshape(-1)
    hf1 = lambda x,_y1=y1: np.linalg.matrix_power(x.reshape(2,2), 2).reshape(-1) - _y1
    x0 = np.random.randn(4)
    x1 = broyden_solver(hf1, x0)
    tmp0 = np.abs(hf1(x1)).max()
    assert tmp0 < 1e-4


def test_func3_complex_value():
    hf1 = lambda x: np.array([x[0]**2+x[1]*x[2]-4.01,x[0]*x[1]+x[1]*x[3]-0.5j,x[0]*x[2]+x[2]*x[3]+0.5j,x[1]*x[2]+x[3]*x[3]-9.01])
    x0 = np.zeros((4,))
    x1 = broyden_solver(hf1, x0, tag_complex=True)
    tmp0 = np.abs(hf1(x1)).max()
    assert tmp0 < 1e-7


def test_np_inv_along_axis12():
    for _ in range(3):
        N0 = np.random.randint(5, 10)
        N1 = np.random.randint(5, 10)
        np1 = np.random.randn(N0,N1,N1) + np.eye(N1)*N1/2
        ret1 = np.stack([np.linalg.inv(x) for x in np1])
        ret2 = np_inv_along_axis12(np1, 1, 2)
        ret3 = np_inv_along_axis12(np1.transpose(2,0,1), 2, 0).transpose(1,2,0)
        assert hfe(ret1, ret2) < 1e-7
        assert hfe(ret1, ret3) < 1e-7
