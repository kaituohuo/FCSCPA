import logging
from types import FunctionType
import numpy as np
import scipy.linalg


def matA_to_Aii(matA:np.ndarray, block_size:int)->np.ndarray:
    assert matA.shape[0]%block_size==0
    num_block = matA.shape[0]//block_size
    return np.diagonal(matA.reshape((num_block,block_size,num_block,block_size)), axis1=0, axis2=2).transpose(2,0,1)


def Aii_to_matA(Aii:np.ndarray)->np.ndarray:
    return scipy.linalg.block_diag(*Aii)


def np_inv_along_axis12(np1:np.ndarray, axis1:int, axis2:int)->np.ndarray:
    ndim = np1.ndim
    assert isinstance(axis1,int) and axis1>=0 and axis1<ndim
    assert isinstance(axis2,int) and axis2>=0 and axis2<ndim
    assert axis1!=axis2
    dim_order = list(range(ndim))
    dim_order.pop(axis2 if axis2>axis1 else axis1)
    dim_order.pop(axis1 if axis2>axis1 else axis2)
    dim_order.append(axis1)
    dim_order.append(axis2)
    dim_order_reverse = [x for x,_ in sorted(enumerate(dim_order), key=lambda x:x[1])]
    return np.linalg.inv(np1.transpose(dim_order)).transpose(dim_order_reverse)


def broyden_solver(func, x0, tag_complex=False, max_iter=1000, max_history=40, beta=0.7, xtol=1e-7, diverge_tol=1e7, return_tag=False):
    '''
    reference: Broyden's method for self-consistent field convergence acceleration
        http://iopscience.iop.org/article/10.1088/0305-4470/17/6/002/meta
    solve problem func(x)=0
    func(callable): (np,float,(N0,)) -> (np,float,(N0,))
    x0(np,float,(N0,))
    max_iter(int)
    max_history(int)
    beta(float)
    xtol(float)
    diverge_tol(float)
    returun_tag(bool)
    (ret0)(np,float,(N0,))
    (ret1)(bool): if return_tag is True
    '''
    N0 = x0.shape[0]
    tmp1 = np.complex if tag_complex else np.float
    delta = np.zeros((max_history,N0), dtype=tmp1)
    Delta = np.zeros((max_history,N0), dtype=tmp1)
    mu = np.zeros((max_history,N0), dtype=tmp1)

    fval0 = func(x0)
    delta_ = beta*fval0
    x1 = x0 + delta_
    fval1 = func(x1)
    tmp1 = np.sqrt(((fval1 - fval0)**2).sum())
    Delta[0] = (fval1-fval0)/tmp1
    delta[0] = delta_/tmp1
    mu[0] = beta*Delta[0] + delta[0]
    num_effective = 1
    x0 = x1
    fval0 = fval1

    tag_success = True
    for ind1 in range(1,max_iter+1):
        ind_cur = ind1%max_history
        if num_effective<max_history:
            # num_effective=ind_cur
            delta_ = beta*fval0 - np.dot(mu[:ind_cur].T, np.dot(Delta[:ind_cur], fval0))
            x1 = x0 + delta_
            fval1 = func(x1)
            tmp1 = np.sqrt(((fval1 - fval0)**2).sum())
            tmp2 = (fval1 - fval0)/tmp1
            tmp3 = np.dot(mu[:ind_cur].T, np.dot(Delta[:ind_cur], tmp2))
        else:
            delta_ = beta*fval0 - np.dot(mu.T, np.dot(Delta, fval0))
            x1 = x0 + delta_
            fval1 = func(x1)
            tmp1 = np.sqrt(((fval1 - fval0)**2).sum())
            tmp2 = (fval1-fval0)/tmp1
            tmp3 = np.dot(mu.T, np.dot(Delta, tmp2))
        Delta[ind_cur] = tmp2
        delta[ind_cur] = delta_/tmp1
        mu[ind_cur] = beta*Delta[ind_cur] + delta[ind_cur] - tmp3
        if np.any(np.isnan(x1)) or np.any(np.abs(x1)>diverge_tol):
            logging.warning('broyden_solver diverge reset value to zero')
            x1 = np.zeros((N0,))
            tag_success = False
            break
        if np.abs(x1-x0).max()<xtol:
            break
        x0 = x1
        fval0 = fval1
        num_effective += 1
    if ind1==max_iter:
        logging.warning('broyden_solver reach max_iter, maybe not converge')
        tag_success = False
    if return_tag:
        return x1, tag_success
    else:
        return x1
