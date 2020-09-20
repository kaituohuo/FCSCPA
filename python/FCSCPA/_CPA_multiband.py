import numpy as np
import scipy.linalg
from types import FunctionType

from FCSCPA._utils import broyden_solver, np_inv_along_axis12

_USE_EINSUM = True


def CPA_block_general(matA:np.ndarray, block_size:int, hf_Bii:FunctionType, hf_average:FunctionType, matC0=None, tag_complex:bool=True)->np.ndarray:
    '''
    matA(np,complex,(N0,N0)): N0%block_size==0, use N1=N0//block_size below
    block_size(int)
    hf_average(callable): (callable,(%)) -> (np,complex,(N1,block_size,block_size))
        (%)(np,float,(block_size,block_size,...)) -> (np,complex,(N1,block_size,block_size,...))
    hf_Bii(callable): (np,?,(...)) -> (np,?,(block_size,block_size,...))
    matC0(np,complex,(N1,block_size,block_size))
    tag_complex(bool)
    (ret)(np,complex,(N1,block_size,block_size))
    '''
    assert matA.shape[0] % block_size==0
    num_block = matA.shape[0]//block_size
    if matC0 is None:
        matC0 = (np.random.rand(num_block, block_size, block_size)-0.5)/10
    else:
        assert matC0.shape==(num_block, block_size, block_size)
    Bii_mean = hf_average(lambda x: hf_Bii(x))
    matA = matA.reshape((num_block,block_size,num_block,block_size)).copy()
    matA[np.arange(num_block),:,np.arange(num_block)] = matA[np.arange(num_block),:,np.arange(num_block)] - Bii_mean
    matA = matA.reshape((num_block*block_size,num_block*block_size))
    hf_Bii = _generate_hf_Bii_without_mean(hf_Bii, Bii_mean)
    hf1 = lambda x: _CPA_block_hf_broyden(x, block_size, matA, hf_Bii, hf_average)
    ret = broyden_solver(hf1, matC0.reshape(-1), tag_complex=tag_complex)
    return ret.reshape((num_block,block_size,block_size)) + Bii_mean

# TODO check it's this necessary
def _generate_hf_Bii_without_mean(hf_Bii, Bii_mean):
    def hf_Bii_without_mean(x):
        ret = hf_Bii(x)
        return ret - Bii_mean.reshape(Bii_mean.shape+(1,)*(ret.ndim-2))
    return hf_Bii_without_mean

def _CPA_block_hf_broyden(x, block_size, matA, hf_Bii, hf_average):
    '''
    x(np,complex,(N0*block_size,))
    (ret)(np,complex,(N0*block_size,))
    '''
    matC = x.reshape((-1,block_size,block_size))
    N1 = matC.shape[0]
    tmp1 = np.linalg.inv(matA - scipy.linalg.block_diag(*matC))
    tmp1 = np.diagonal(tmp1.reshape((N1,block_size,N1,block_size)), axis2=2).transpose((2,0,1))
    ret = hf_average(lambda y: _CPA_block_hf_average(hf_Bii(y), tmp1, matC)).reshape(-1) - x
    return ret


def _CPA_block_hf_average(y, Gii, matC):
    '''
    y(np,?,(block_size,block_size,...))
    Gii(np,complex/float,(N1,block_size,block_size))
    matC(np,complex/float,(N1,block_size,block_size))
    (ret)(np,complex/float,(N1,block_size,block_size,...))
    '''
    num_block,block_size,_ = matC.shape
    assert isinstance(y, np.ndarray) and y.ndim>=2 and y.shape[0]==block_size and y.shape[1]==block_size
    eye1 = np.eye(block_size)
    if y.ndim>2:
        tmp1 = (1,)*(y.ndim-2)
        eye1 = eye1.reshape((block_size,block_size) + tmp1)
        Gii = Gii.reshape((num_block,block_size,block_size) + tmp1)
        matC = matC.reshape((num_block,block_size,block_size) + tmp1)
    if _USE_EINSUM:
        tmp1 = eye1 - np.einsum(Gii, [0,1,2,...], y-matC, [0,2,3,...], [0,1,3,...], optimize=True)
        ret = np.einsum(y, [1,2,...], np_inv_along_axis12(tmp1, axis1=1, axis2=2), [0,2,3,...], [0,1,3,...], optimize=True)
    else:
        tmp1 = eye1 - np.sum(Gii[:,:,:,None]*(y-matC[:,None]), axis=2)
        ret = np.sum(y[:,:,None] * np_inv_along_axis12(tmp1, axis1=1, axis2=2)[:,None], axis=2)
    return ret
