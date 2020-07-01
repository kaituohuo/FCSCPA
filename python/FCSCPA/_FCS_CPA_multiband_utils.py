import numpy as np
import scipy.linalg
from types import FunctionType

from FCSCPA._CPA_multiband import CPA_block_general
from FCSCPA._utils import np_inv_along_axis12

_USE_EINSUM = False #terrible performance to use "np.einsum" even for (num_block=24, block_size=3), maybe due to optimize=True

def hf_dot_matmul(*args):
    '''equivalent with
    lambda x0,x1: np.einsum(x0, [0,1,2,...], x1, [0,2,3,...], [0,1,3,...])
    lambda x0,x1,x2: np.einsum(x0, [0,1,2,...], x1, [0,2,3,...], x2, [0,3,4,...], [0,1,4,...])
    lambda x0,x1,x2,x3: np.einsum(x0, [0,1,2,...], x1, [0,2,3,...], x2, [0,3,4,...], x3, [0,4,5,...], [0,1,5,...])
    lambda x0,x1,x2,x3,x4: np.einsum(x0, [0,1,2,...], x1, [0,2,3,...], x2, [0,3,4,...], x3, [0,4,5,...], x4, [0,5,6,...], [0,1,6,...])
    '''
    num1 = len(args)
    assert num1>=2 and all(isinstance(x, np.ndarray) for x in args)
    tmp1 = [[0,x,y,...] for x,y in zip(range(1, num1+1), range(2,num1+2))]
    return np.einsum(*[y for x in zip(args,tmp1) for y in x], [0,1,num1+1,...], optimize=True)

# TODO for time profile
# import time
# def zct_print(note, zct0):
#     zct1 = time.time()
#     print('[{}]: \t {}'.format(note, zct1-zct0))
#     return zct1

def FCS_CPA_multiband(num_cumulant:int, inv_Green_part:np.ndarray,
        Gamma12:np.ndarray, Gamma21:np.ndarray, hf_Bii:FunctionType, hf_average:FunctionType)->np.ndarray:
    '''documentation see FCS_CPA'''
    NBPA = hf_Bii(0).shape[0]
    assert inv_Green_part.shape[0]%NBPA==0
    num_atom = inv_Green_part.shape[0]//NBPA
    N_list = [[None,None] for _ in range(2*num_cumulant)]
    Nii_list = [[None,None] for _ in range(2*num_cumulant)]
    Delta_list = [[None,None] for _ in range(2*num_cumulant)]

    Delta_list[0][0] = CPA_block_general(inv_Green_part, NBPA, hf_Bii, hf_average)
    Delta_list[0][1] = np.conjugate((Delta_list[0][0].transpose(0,2,1)))
    # Delta_list[0][1] = CPA_block_general(inv_Green_part.T.conj(), NBPA, hf_Bii, hf_average)
    N_list[0][0] = np.linalg.inv(inv_Green_part - scipy.linalg.block_diag(*Delta_list[0][0]))
    # N_list[0][1] = N_list[0][0].T.conj()
    N_list[0][1] = np.linalg.inv(inv_Green_part.T.conj() - scipy.linalg.block_diag(*Delta_list[0][1]))
    Nii_list[0][0] = np.diagonal(N_list[0][0].reshape((num_atom,NBPA,num_atom,NBPA)), axis1=0, axis2=2).transpose(2,0,1)
    Nii_list[0][1] = np.diagonal(N_list[0][1].reshape((num_atom,NBPA,num_atom,NBPA)), axis1=0, axis2=2).transpose(2,0,1)

    BKBK_11,BKBK_12,BKBK_21,BKBK_22, BKK_11,BKK_12,BKK_21,BKK_22 = \
                get_BKBK_BKK(Delta_list[0][0], Delta_list[0][1], Nii_list[0][0], Nii_list[0][1], hf_Bii, hf_average)
    eye1 = np.eye(num_atom*NBPA*NBPA).reshape((num_atom,NBPA,NBPA,num_atom,NBPA,NBPA))
    N0ij_11 = N_list[0][0].reshape(num_atom,NBPA,num_atom,NBPA).copy()
    N0ij_11[np.arange(num_atom),:,np.arange(num_atom)] = 0
    N0ij_22 = N_list[0][1].reshape(num_atom,NBPA,num_atom,NBPA).copy()
    N0ij_22[np.arange(num_atom),:,np.arange(num_atom)] = 0
    A11 = get_Aii(eye1, BKBK_11, N0ij_11, N0ij_11)
    A12 = get_Aii(eye1, BKBK_12, N0ij_11, N0ij_22)
    A21 = get_Aii(eye1, BKBK_21, N0ij_22, N0ij_11)
    A22 = get_Aii(eye1, BKBK_22, N0ij_22, N0ij_22)

    tmp1 = N_list[0][0].reshape((num_atom,NBPA,num_atom,NBPA))
    tmp2 = N_list[0][1].reshape((num_atom,NBPA,num_atom,NBPA))
    tmp3 = Gamma12.reshape((num_atom,NBPA,num_atom,NBPA))
    tmp4 = Gamma21.reshape((num_atom,NBPA,num_atom,NBPA))
    B1_12 = get_B1(BKBK_12, tmp1, tmp2, tmp3)
    B1_21 = get_B1(BKBK_21, tmp2, tmp1, tmp4)
    # B1_21 = - np.einsum(BKBK_21, [0,1,2,3,4], tmp2, [0,3,5,6], tmp4, [5,6,7,8], tmp1, [7,8,0,4], [0,1,2], optimize=True)
    Delta_list[1][0] = np.linalg.solve(A12, B1_12.reshape(-1)).reshape((num_atom,NBPA,NBPA))
    Delta_list[1][1] = np.linalg.solve(A21, B1_21.reshape(-1)).reshape((num_atom,NBPA,NBPA))

    N_list[1][0] = N_list[0][0] @ (scipy.linalg.block_diag(*Delta_list[1][0]) - Gamma12) @ N_list[0][1]
    N_list[1][1] = N_list[0][1] @ (scipy.linalg.block_diag(*Delta_list[1][1]) - Gamma21) @ N_list[0][0]
    Nii_list[1][0] = N_list[1][0].reshape((num_atom,NBPA,num_atom,NBPA))[np.arange(num_atom),:,np.arange(num_atom)]
    Nii_list[1][1] = N_list[1][1].reshape((num_atom,NBPA,num_atom,NBPA))[np.arange(num_atom),:,np.arange(num_atom)]

    for ind1 in range(2, 2*num_cumulant):
        tmp1,tmp2 = get_matB(N_list[:ind1], Nii_list[:ind1], Delta_list[:ind1], Gamma12, Gamma21,
                hf_Bii, hf_average, (BKBK_11,BKBK_12,BKBK_21,BKBK_22), (BKK_11,BKK_12,BKK_21,BKK_22))
        Delta_list[ind1][0] = np.linalg.solve(A12 if ind1%2 else A11, tmp1.reshape(-1)).reshape((num_atom,NBPA,NBPA))
        Delta_list[ind1][1] = np.linalg.solve(A21 if ind1%2 else A22, tmp2.reshape(-1)).reshape((num_atom,NBPA,NBPA))

        tmp1 = [x[0] for x in Delta_list[1:(ind1+1)]]
        tmp2 = [x[y%2].reshape((num_atom,NBPA,num_atom,NBPA)) for x,y in zip(N_list[(ind1-1)::-1],range(1,ind1+1))]
        tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3,4], [0,1,3,4], optimize=True) for x,y in zip(tmp1,tmp2)).reshape((num_atom*NBPA,-1))
        N_list[ind1][0] = N_list[0][0] @ (tmp3 - Gamma12 @ N_list[ind1-1][1])
        tmp1 = [x[1] for x in Delta_list[1:(ind1+1)]]
        tmp2 = [x[(y+1)%2].reshape((num_atom,NBPA,num_atom,NBPA)) for x,y in zip(N_list[(ind1-1)::-1],range(1,ind1+1))]
        tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3,4], [0,1,3,4], optimize=True) for x,y in zip(tmp1,tmp2)).reshape((num_atom*NBPA,-1))
        N_list[ind1][1] = N_list[0][1] @ (tmp3 - Gamma21 @ N_list[ind1-1][0])
        Nii_list[ind1][0] = N_list[ind1][0].reshape((num_atom,NBPA,num_atom,NBPA))[np.arange(num_atom),:,np.arange(num_atom)]
        Nii_list[ind1][1] = N_list[ind1][1].reshape((num_atom,NBPA,num_atom,NBPA))[np.arange(num_atom),:,np.arange(num_atom)]
    return np.array([(x*Gamma21.T).sum() + (y*Gamma12.T).sum() for x,y in N_list[1::2]])

def get_BKBK_BKK(Delta0_11, Delta0_22, N0ii_11, N0ii_22, hf_Bii, hf_average):
    hf1 = lambda x: hf_BKBK_BKK(hf_Bii(x), Delta0_11, Delta0_22, N0ii_11, N0ii_22)
    return hf_average(hf1)

def hf_BKBK_BKK(x, Delta0_11, Delta0_22, N0ii_11, N0ii_22):
    '''
    x(np,?,(block_size,block_size,...))
    Delta0_11/Delta0_22/N0ii_11/N0ii_22(np,?,(num_atom,block_size,block_size))
    '''
    assert isinstance(x, np.ndarray) and x.ndim>=2
    eye1 = np.eye(Delta0_11.shape[1])
    if x.ndim>2:
        tmp1 = Delta0_11.shape + (1,)*(x.ndim-2)
        Delta0_11 = Delta0_11.reshape(tmp1)
        Delta0_22 = Delta0_22.reshape(tmp1)
        N0ii_11 = N0ii_11.reshape(tmp1)
        N0ii_22 = N0ii_22.reshape(tmp1)
        eye1 = eye1.reshape(eye1.shape+(1,)*(x.ndim-2))
    B11 = Delta0_11 - x
    B22 = Delta0_22 - x
    K0ii_11 = np_inv_along_axis12(eye1 + hf_dot_matmul(N0ii_11, B11), 1, 2)
    K0ii_22 = np_inv_along_axis12(eye1 + hf_dot_matmul(N0ii_22, B22), 1, 2)
    BK_11 = hf_dot_matmul(B11, K0ii_11)
    BK_22 = hf_dot_matmul(B22, K0ii_22)
    BKBK_11 = np.einsum(BK_11, [0,1,2,...], BK_11, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKBK_12 = np.einsum(BK_11, [0,1,2,...], BK_22, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKBK_21 = np.einsum(BK_22, [0,1,2,...], BK_11, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKBK_22 = np.einsum(BK_22, [0,1,2,...], BK_22, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKK_11 = np.einsum(BK_11, [0,1,2,...], K0ii_11, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKK_12 = np.einsum(BK_11, [0,1,2,...], K0ii_22, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKK_21 = np.einsum(BK_22, [0,1,2,...], K0ii_11, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    BKK_22 = np.einsum(BK_22, [0,1,2,...], K0ii_22, [0,3,4,...], [0,1,4,2,3,...], optimize=True)
    return np.stack([BKBK_11, BKBK_12, BKBK_21, BKBK_22, BKK_11, BKK_12, BKK_21, BKK_22], axis=0)


def get_matB(N_list, Nii_list, Delta_list, Gamma12, Gamma21, hf_Bii, hf_average, BKBK, BKK):
    assert len(Delta_list)==len(Nii_list) and len(Delta_list)==len(N_list)
    ind1 = len(Nii_list)
    num_atom,NBPA,_ = Delta_list[0][0].shape
    BKBK_11,BKBK_12,BKBK_21,BKBK_22 = BKBK
    BKBK_1 = BKBK_11 if (ind1%2==0) else BKBK_12
    BKBK_2 = BKBK_22 if (ind1%2==0) else BKBK_21
    BKK_11,BKK_12,BKK_21,BKK_22 = BKK
    BKK_1 = BKK_11 if (ind1%2==0) else BKK_12
    BKK_2 = BKK_22 if (ind1%2==0) else BKK_21
    hf_reshape = lambda x: x.reshape((num_atom,NBPA,num_atom,NBPA))

    part1_0,part1_1 = hf_average(lambda x: hf_matB_integral(hf_Bii(x), Nii_list, Delta_list))

    tmp1 = [x[0] for x in Nii_list[1:ind1]]
    tmp2 = [x[y%2] for x,y in zip(Delta_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3], [0,1,3], optimize=True) for x,y in zip(tmp1,tmp2))
    part2_0 = np.einsum(BKK_1, [0,1,2,3,4], tmp3, [0,3,4], [0,1,2], optimize=True)
    tmp1 = [x[1] for x in Nii_list[1:ind1]]
    tmp2 = [x[(y+1)%2] for x,y in zip(Delta_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3], [0,1,3], optimize=True) for x,y in zip(tmp1,tmp2))
    part2_1 = np.einsum(BKK_2, [0,1,2,3,4], tmp3, [0,3,4], [0,1,2], optimize=True)

    tmp1 = [x[0] for x in Delta_list[1:ind1]]
    tmp2 = [hf_reshape(x[y%2]) for x,y in zip(N_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3,4], [0,1,3,4], optimize=True) for x,y in zip(tmp1,tmp2))
    part3_0 = get_matB_part3(BKBK_1, hf_reshape(N_list[0][0]), tmp3)
    tmp1 = [x[1] for x in Delta_list[1:ind1]]
    tmp2 = [hf_reshape(x[(y+1)%2]) for x,y in zip(N_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(np.einsum(x, [0,1,2], y, [0,2,3,4], [0,1,3,4], optimize=True) for x,y in zip(tmp1,tmp2))
    part3_1 = get_matB_part3(BKBK_2, hf_reshape(N_list[0][1]), tmp3)

    part4_0 = get_matB_part4(BKBK_1, hf_reshape(N_list[0][0]), hf_reshape(Gamma12), hf_reshape(N_list[ind1-1][1]))
    part4_1 = get_matB_part4(BKBK_2, hf_reshape(N_list[0][1]), hf_reshape(Gamma21), hf_reshape(N_list[ind1-1][0]))

    return part1_0+part2_0+part3_0+part4_0, part1_1+part2_1+part3_1+part4_1


def hf_matB_integral(x, Nii_list, Delta_list):
    assert isinstance(x, np.ndarray) and x.ndim>=2

    ind1 = len(Nii_list)
    NBPA = Delta_list[0][0].shape[1]
    eye1 = np.eye(NBPA)
    if x.ndim>2:
        hf1 = lambda x,_ndim=(x.ndim-2): x.reshape(x.shape+(1,)*_ndim)
        Nii_list = [[hf1(x),hf1(y)] for x,y in Nii_list]
        Delta_list = [[hf1(x),hf1(y)] for x,y in Delta_list]
        eye1 = hf1(eye1)

    Hii_list = [[None,None] for _ in range(ind1)]
    Hii_list[0][0] = eye1 + hf_dot_matmul(Nii_list[0][0], Delta_list[0][0]-x)
    Hii_list[0][1] = eye1 + hf_dot_matmul(Nii_list[0][1], Delta_list[0][1]-x)
    for ind2 in range(1,ind1):
        tmp1 = [x[0] for x in Nii_list[:ind2]]
        tmp2 = [x[y%2] for x,y in zip(Delta_list[ind2:0:-1],range(0,ind2))]
        Hii_list[ind2][0] = sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2)) + hf_dot_matmul(Nii_list[ind2][0], (Delta_list[0][ind2%2]-x))
        tmp1 = [x[1] for x in Nii_list[:ind2]]
        tmp2 = [x[(y+1)%2] for x,y in zip(Delta_list[ind2:0:-1],range(0,ind2))]
        Hii_list[ind2][1] = sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2)) + hf_dot_matmul(Nii_list[ind2][1], (Delta_list[0][(ind2+1)%2]-x))

    Kii_list = [[None,None] for _ in range(ind1)]
    Kii_list[0][0] = np_inv_along_axis12(Hii_list[0][0], 1, 2)
    Kii_list[0][1] = np_inv_along_axis12(Hii_list[0][1], 1, 2)
    for ind2 in range(1,ind1):
        tmp1 = [x[0] for x in Hii_list[1:ind2]]
        tmp2 = [x[y%2] for x,y in zip(Kii_list[(ind2-1):0:-1],range(1,ind2))]
        tmp3 = 0 if len(tmp1)==0 else hf_dot_matmul(Kii_list[0][0], sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2)))
        Kii_list[ind2][0] = - tmp3 - Kii_list[0][0]*Hii_list[ind2][0]*Kii_list[0][ind2%2]
        tmp1 = [x[1] for x in Hii_list[1:ind2]]
        tmp2 = [x[(y+1)%2] for x,y in zip(Kii_list[(ind2-1):0:-1],range(1,ind2))]
        tmp3 = 0 if len(tmp1)==0 else hf_dot_matmul(Kii_list[0][1], sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2)))
        Kii_list[ind2][1] = - tmp3 - hf_dot_matmul(Kii_list[0][1], Hii_list[ind2][1], Kii_list[0][(ind2+1)%2])

    part1 = [None,None]
    tmp1 = [x[0] for x in Hii_list[1:ind1]]
    tmp2 = [x[y%2] for x,y in zip(Kii_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2))
    part1[0] = hf_dot_matmul(Delta_list[0][0]-x, Kii_list[0][0], tmp3)
    tmp1 = [x[1] for x in Hii_list[1:ind1]]
    tmp2 = [x[(y+1)%2] for x,y in zip(Kii_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = sum(hf_dot_matmul(x,y) for x,y in zip(tmp1,tmp2))
    part1[1] = hf_dot_matmul(Delta_list[0][1]-x, Kii_list[0][1], tmp3)

    return np.stack([part1[0], part1[1]], axis=0)


# WARNING, all function below are just for speed
# the code in "_USE_EINSUM" block should always give the SAME results but sooooo slow
def get_Aii(eye1, BKBK, first_N0ij, second_N0ij):
    num_atom,NBPA = eye1.shape[:2]
    assert eye1.shape == (num_atom, NBPA, NBPA, num_atom, NBPA, NBPA)
    assert BKBK.shape == (num_atom, NBPA, NBPA, NBPA, NBPA)
    assert first_N0ij.shape == (num_atom, NBPA, num_atom, NBPA)
    assert second_N0ij.shape == (num_atom, NBPA, num_atom, NBPA)
    if _USE_EINSUM:
        ret = (eye1 - np.einsum(BKBK, [0,1,2,3,4], first_N0ij, [0,3,5,6], second_N0ij, [5,7,0,4],
                [0,1,2,5,6,7], optimize=True)).reshape((num_atom*NBPA*NBPA,-1))
    else:
        tmp1 = np.sum(BKBK.reshape(BKBK.shape+(1,1)) * first_N0ij.reshape((num_atom,1,1,NBPA,1,num_atom,NBPA)), axis=3)
        tmp2 = np.sum(tmp1.reshape(tmp1.shape+(1,)) * second_N0ij.reshape((num_atom,1,1,NBPA,num_atom,1,NBPA)), axis=3)
        ret = (eye1 - tmp2).reshape((num_atom*NBPA*NBPA,-1))
    return ret

def get_B1(BKBK, para1, para2, para3):
    num_atom,NBPA = BKBK.shape[:2]
    assert BKBK.shape == (num_atom, NBPA, NBPA, NBPA, NBPA)
    assert para1.shape == (num_atom, NBPA, num_atom, NBPA)
    assert para3.shape == (num_atom, NBPA, num_atom, NBPA)
    assert para3.shape == (num_atom, NBPA, num_atom, NBPA)
    if _USE_EINSUM:
        ret = - np.einsum(BKBK, [0,1,2,3,4], para1, [0,3,5,6], para3, [5,6,7,8], para2, [7,8,0,4], [0,1,2], optimize=True)
    else:
        tmp1 = (para1.reshape((num_atom*NBPA, num_atom*NBPA)) @ para3.reshape((num_atom*NBPA, num_atom*NBPA))).reshape((num_atom, NBPA, 1, num_atom*NBPA))
        tmp2 = np.sum(tmp1 * (para2.transpose(2,3,0,1).reshape((num_atom, 1, NBPA, -1))), axis=-1)
        ret = -np.sum((BKBK.reshape(BKBK.shape[:3]+(-1,))) * (tmp2.reshape((num_atom,1,1,-1))), axis=-1)
    return ret

def get_matB_part3(BKBK, N0, para3):
    num_atom,NBPA = BKBK.shape[:2]
    assert BKBK.shape == (num_atom, NBPA, NBPA, NBPA, NBPA)
    assert N0.shape == (num_atom,NBPA,num_atom,NBPA)
    assert para3.shape == (num_atom,NBPA,num_atom,NBPA)
    if _USE_EINSUM:
        ret = np.einsum(BKBK, [0,1,2,3,4], N0, [0,3,5,6], para3, [5,6,0,4], [0,1,2], optimize=True)
    else:
        tmp1 = np.sum(N0.reshape((num_atom,NBPA,1,-1)) * para3.transpose(2,3,0,1).reshape((num_atom,1,NBPA,-1)), axis=3).reshape((num_atom,1,1,NBPA*NBPA))
        ret = np.sum(BKBK.reshape((num_atom,NBPA,NBPA,-1)) * tmp1, axis=3)
    return ret

def get_matB_part4(BKBK, N0, Gamma, Nk):
    num_atom, NBPA = BKBK.shape[:2]
    assert BKBK.shape == (num_atom, NBPA, NBPA, NBPA, NBPA)
    assert N0.shape == (num_atom, NBPA, num_atom, NBPA)
    assert Gamma.shape == (num_atom, NBPA, num_atom, NBPA)
    assert Nk.shape == (num_atom, NBPA, num_atom, NBPA)
    if _USE_EINSUM:
        ret = - np.einsum(BKBK, [0,1,2,3,4], N0, [0,3,5,6], Gamma, [5,6,7,8], Nk, [7,8,0,4], [0,1,2], optimize=True)
    else:
        tmp1 = (N0.reshape((num_atom*NBPA,-1)) @ Gamma.reshape((num_atom*NBPA,-1))).reshape((num_atom,NBPA,1,-1))
        tmp2 = np.sum(tmp1 * Nk.transpose(2,3,0,1).reshape((num_atom,1,NBPA,-1)), axis=3).reshape((num_atom, 1, 1, -1))
        ret = - np.sum(BKBK.reshape(num_atom,NBPA,NBPA,-1)*tmp2, axis=3)
    return ret
