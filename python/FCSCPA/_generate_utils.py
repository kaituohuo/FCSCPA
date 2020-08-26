import numpy as np
from types import FunctionType
import quadpy

def generate_hf_Bii(mat1:np.ndarray)->FunctionType:
    def hf_Bii(x):
        assert isinstance(x, (np.ndarray,int,float,complex))
        if isinstance(x, np.ndarray):
            return x * mat1.reshape(mat1.shape + (1,)*x.ndim)
        else:
            return x * mat1
    return hf_Bii

def generate_uniform_hf_rand(x1:float, x2:float)->FunctionType:
    assert x2 > x1
    def hf_rand(*size):
        return np.random.rand(*size) * (x2-x1) + x1
    return hf_rand

def generate_uniform_hf_average(x1:float, x2:float)->FunctionType:
    assert x2 > x1
    def hf_average(hf1):
        return quadpy.c1.integrate_adaptive(hf1, [x1,x2], 1e-5)[0]/(x2-x1) #TODO
    return hf_average

def generate_binary_hf_rand(x1:float, c1:float, x2:float)->FunctionType:
    assert c1>=0 and c1<=1
    tmp1 = np.array([x1, x2])
    tmp2 = np.array([c1, 1-c1])
    def hf_rand(*size):
        return np.random.choice(tmp1, size=size, p=tmp2)
    return hf_rand

def generate_binary_hf_average(x1:float, c1:float, x2:float)->FunctionType:
    def hf_average(hf1):
        return hf1(x1)*c1 + hf1(x2)*(1-c1)
    return hf_average
