import numpy as np
import scipy.linalg
from types import FunctionType

from FCSCPA._utils import broyden_solver

def CPA_general(matA:np.ndarray, hf_average:FunctionType, matC0=None, tag_complex:bool=True)->np.ndarray:
    '''
    matA(np,complex,(N0,N0))
    hf_average(callable): (callable,(%)) -> (np,complex,(N0,))
        (%)(np,float,(...)) -> (np,complex,(N0,...))
    matC0(np,complex,(N0,))
    tag_complex(bool)
    (ret)(np,complex,(N0,))
    '''
    if matC0 is None:
        matC0 = (np.random.rand(matA.shape[0])-0.5)/10
    Bii_mean = hf_average(lambda x: x)
    matA = matA - Bii_mean*np.eye(matA.shape[0])
    hf1 = lambda x: _CPA_hf_broyden(x, matA, Bii_mean, hf_average)
    return broyden_solver(hf1, matC0, tag_complex=tag_complex) + Bii_mean


def _CPA_hf_broyden(x, matA, Bii_mean, hf_average):
    '''
    x(np,complex,(N0,)): matC
    (ret)(np,complex,(N0,))
    '''
    Gii = np.diag(np.linalg.inv(matA - np.diag(x)))
    return hf_average(lambda y: _CPA_hf_average(y-Bii_mean, Gii, x)) - x

def _CPA_hf_average(y, Gii, matC):
    '''
    y(np,?,(...))
    y(float)
    y(int)
    y(complex)
    Gii(np,complex/float,(N0,))
    matC(np,complex/float,(N0,))
    (ret)(np,complex/float,(N0,...))
    '''
    assert isinstance(y, (int,float,complex,np.ndarray))
    if isinstance(y, np.ndarray):
        tmp1 = Gii.shape + (1,)*y.ndim
        return y/(1-Gii.reshape(tmp1)*(y-matC.reshape(tmp1)))
    else:
        return y/(1-Gii*(y-matC))
