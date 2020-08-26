import numpy as np
from types import FunctionType

from FCSCPA._CPA_oneband import CPA_general

hfH = lambda x: np.conjugate(x.T)

def FCS_CPA_oneband(num_cumulant:int, inv_Green_part:np.ndarray,
        Gamma12:np.ndarray, Gamma21:np.ndarray,
        hf_average:FunctionType, matC0:np.ndarray=None)->np.ndarray:
    '''see FCS_CPA
    abbreviation:
        N0(num_cumulant)
        N1(num_atom)

    num_cumulant(int):
    hf_average(FunctionType):
        hf1(x): the last two dimension is kept for quadpy [WARNING]
    inv_Green_part(np,complex,(N0*N1,N0*N1))
    Gamma12(np,complex,(N0*N1,N0*N1))
    Gamma21(np,complex,(N0*N1,N0*N1))
    '''
    num_atom = inv_Green_part.shape[0]
    N_list = [[None,None] for _ in range(2*num_cumulant)]
    Nii_list = [[None,None] for _ in range(2*num_cumulant)]
    Delta_list = [[None,None] for _ in range(2*num_cumulant)]

    matC0 = CPA_general(inv_Green_part, hf_average, matC0)
    Delta_list[0][0] = matC0
    Delta_list[0][1] = hfH(Delta_list[0][0])
    N_list[0][0] = np.linalg.inv(inv_Green_part - np.diag(Delta_list[0][0]))
    N_list[0][1] = hfH(N_list[0][0])
    Nii_list[0][0] = np.diag(N_list[0][0])
    Nii_list[0][1] = np.diag(N_list[0][1])

    BKBK_11,BKBK_12,BKBK_21,BKBK_22,BKK_11,BKK_12,BKK_21,BKK_22 = \
                get_BKBK_BKK(Delta_list[0][0], Delta_list[0][1], Nii_list[0][0], Nii_list[0][1], hf_average)

    eye1 = np.eye(num_atom)
    N0ij_11 = N_list[0][0].copy()
    N0ij_11[np.arange(num_atom), np.arange(num_atom)] = 0
    N0ij_22 = N_list[0][1].copy()
    N0ij_22[np.arange(num_atom), np.arange(num_atom)] = 0
    A11 = eye1 - BKBK_11[:,np.newaxis] * (N0ij_11 * N0ij_11.T)
    A12 = eye1 - BKBK_12[:,np.newaxis] * (N0ij_11 * N0ij_22.T)
    A21 = eye1 - BKBK_21[:,np.newaxis] * (N0ij_22 * N0ij_11.T)
    A22 = eye1 - BKBK_22[:,np.newaxis] * (N0ij_22 * N0ij_22.T)

    tmp1 = ((N_list[0][0]@Gamma12)*N_list[0][1].T).sum(axis=1)
    Delta_list[1][0] = -np.linalg.solve(A12, BKBK_12*tmp1)
    tmp1 = ((N_list[0][1]@Gamma21)*N_list[0][0].T).sum(axis=1)
    Delta_list[1][1] = -np.linalg.solve(A21, BKBK_21*tmp1)

    N_list[1][0] = N_list[0][0] @ (np.diag(Delta_list[1][0])-Gamma12) @ N_list[0][1]
    N_list[1][1] = N_list[0][1] @ (np.diag(Delta_list[1][1])-Gamma21) @ N_list[0][0]
    Nii_list[1][0] = np.diag(N_list[1][0])
    Nii_list[1][1] = np.diag(N_list[1][1])

    for ind1 in range(2, 2*num_cumulant):
        tmp1,tmp2 = get_matB(N_list[:ind1], Nii_list[:ind1], Delta_list[:ind1], Gamma12, Gamma21,
                hf_average, [BKBK_11,BKBK_12,BKBK_21,BKBK_22], [BKK_11,BKK_12,BKK_21,BKK_22])
        Delta_list[ind1][0] = np.linalg.solve(A12 if ind1%2 else A11, tmp1)
        Delta_list[ind1][1] = np.linalg.solve(A21 if ind1%2 else A22, tmp2)

        tmp1 = [x[0] for x in Delta_list[1:(ind1+1)]]
        tmp2 = [x[y%2] for x,y in zip(N_list[(ind1-1)::-1],range(1,ind1+1))]
        N_list[ind1][0] = N_list[0][0] @ (sum(x[:,np.newaxis]*y for x,y in zip(tmp1,tmp2)) - Gamma12 @ N_list[ind1-1][1])
        Nii_list[ind1][0] = np.diag(N_list[ind1][0])
        tmp1 = [x[1] for x in Delta_list[1:(ind1+1)]]
        tmp2 = [x[(y+1)%2] for x,y in zip(N_list[(ind1-1)::-1],range(1,ind1+1))]
        N_list[ind1][1] = N_list[0][1] @ (sum(x[:,np.newaxis]*y for x,y in zip(tmp1,tmp2)) - Gamma21 @ N_list[ind1-1][0])
        Nii_list[ind1][1] = np.diag(N_list[ind1][1])
    N_Gamma_trace = np.array([(x*Gamma21.T).sum() + (y*Gamma12.T).sum() for x,y in N_list[1::2]])
    return N_Gamma_trace, matC0


def get_BKBK_BKK(Delta0_11, Delta0_22, N0ii_11, N0ii_22, hf_average):
    hf1 = lambda x: hf_BKBK_BKK_integral(x, Delta0_11, Delta0_22, N0ii_11, N0ii_22)
    return hf_average(hf1)


def hf_BKBK_BKK_integral(x, Delta0_11, Delta0_22, N0ii_11, N0ii_22):
    if isinstance(x, np.ndarray):
        tmp1 = Delta0_11.shape + (1,)*x.ndim
        Delta0_11 = Delta0_11.reshape(tmp1)
        Delta0_22 = Delta0_22.reshape(tmp1)
        N0ii_11 = N0ii_11.reshape(tmp1)
        N0ii_22 = N0ii_22.reshape(tmp1)
    B_11 = Delta0_11 - x
    B_22 = Delta0_22 - x
    K0_11 = 1/(1 + N0ii_11*B_11)
    K0_22 = 1/(1 + N0ii_22*B_22)
    BK_11 = B_11 * K0_11
    BK_22 = B_22 * K0_22
    BKBK_11 = BK_11 * BK_11
    BKBK_12 = BK_11 * BK_22
    BKBK_21 = BK_22 * BK_11
    BKBK_22 = BK_22 * BK_22
    BKK_11 = BK_11 * K0_11
    BKK_12 = BK_11 * K0_22
    BKK_21 = BK_22 * K0_11
    BKK_22 = BK_22 * K0_22
    return np.stack([BKBK_11,BKBK_12,BKBK_21,BKBK_22,BKK_11,BKK_12,BKK_21,BKK_22])


def get_matB(N_list, Nii_list, Delta_list, Gamma12, Gamma21, hf_average, BKBK, BKK):
    assert len(Delta_list)==len(Nii_list) and len(Delta_list)==len(N_list)
    hf_diag = lambda x,y: (x*y.T).sum(axis=1)
    ind1 = len(Nii_list)
    BKBK_11,BKBK_12,BKBK_21,BKBK_22 = BKBK
    BKK_11,BKK_12,BKK_21,BKK_22 = BKK

    part1_0,part1_1 = hf_average(lambda x: hf_matB_integral(x, Nii_list, Delta_list, Gamma12, Gamma21))

    tmp1 = [x[0] for x in Nii_list[1:ind1]]
    tmp2 = [x[y%2] for x,y in zip(Delta_list[(ind1-1):0:-1],range(1,ind1))]
    part2_0 = (BKK_11 if (ind1%2==0) else BKK_12) * sum(x*y for x,y in zip(tmp1,tmp2))
    tmp1 = [x[1] for x in Nii_list[1:ind1]]
    tmp2 = [x[(y+1)%2] for x,y in zip(Delta_list[(ind1-1):0:-1],range(1,ind1))]
    part2_1 = (BKK_22 if (ind1%2==0) else BKK_21) * sum(x*y for x,y in zip(tmp1,tmp2))

    tmp1 = [x[0] for x in Delta_list[1:ind1]]
    tmp2 = [x[y%2] for x,y in zip(N_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = hf_diag(N_list[0][0], sum(x[:,np.newaxis]*y for x,y in zip(tmp1,tmp2)))
    part3_0 = (BKBK_11 if (ind1%2==0) else BKBK_12) * tmp3
    tmp1 = [x[1] for x in Delta_list[1:ind1]]
    tmp2 = [x[(y+1)%2] for x,y in zip(N_list[(ind1-1):0:-1],range(1,ind1))]
    tmp3 = hf_diag(N_list[0][1], sum(x[:,np.newaxis]*y for x,y in zip(tmp1,tmp2)))
    part3_1 = (BKBK_22 if (ind1%2==0) else BKBK_21) * tmp3

    part4_0 = - (BKBK_11 if (ind1%2==0) else BKBK_12) * hf_diag(N_list[0][0] @ Gamma12, N_list[ind1-1][1])
    part4_1 = - (BKBK_22 if (ind1%2==0) else BKBK_21) * hf_diag(N_list[0][1] @ Gamma21, N_list[ind1-1][0])

    return part1_0+part2_0+part3_0+part4_0, part1_1+part2_1+part3_1+part4_1


# TODO DeltaVQ
def hf_matB_integral(x, Nii_list, Delta_list, Gamma12, Gamma21):
    ind1 = len(Nii_list)
    if isinstance(x, np.ndarray):
        hf1 = lambda x,_ndim=x.ndim: x.reshape(x.shape+(1,)*_ndim)
        Nii_list = [[hf1(x),hf1(y)] for x,y in Nii_list]
        Delta_list = [[hf1(x),hf1(y)] for x,y in Delta_list]

    Hii_list = [[None,None] for _ in range(ind1)]
    Hii_list[0][0] = 1 - Nii_list[0][0]*(x - Delta_list[0][0])
    Hii_list[0][1] = 1 - Nii_list[0][1]*(x - Delta_list[0][1])
    for ind2 in range(1,ind1):
        tmp1 = [x[0] for x in Nii_list[:ind2]]
        tmp2 = [x[y%2] for x,y in zip(Delta_list[ind2:0:-1],range(0,ind2))]
        Hii_list[ind2][0] = sum(x*y for x,y in zip(tmp1,tmp2)) - Nii_list[ind2][0]*(x-Delta_list[0][ind2%2])
        tmp1 = [x[1] for x in Nii_list[:ind2]]
        tmp2 = [x[(y+1)%2] for x,y in zip(Delta_list[ind2:0:-1],range(0,ind2))]
        Hii_list[ind2][1] = sum(x*y for x,y in zip(tmp1,tmp2)) - Nii_list[ind2][1]*(x-Delta_list[0][(ind2+1)%2])

    Kii_list = [[None,None] for _ in range(ind1)]
    Kii_list[0][0] = 1/Hii_list[0][0]
    Kii_list[0][1] = 1/Hii_list[0][1]
    for ind2 in range(1,ind1):
        tmp1 = [x[0] for x in Hii_list[1:ind2]]
        tmp2 = [x[y%2] for x,y in zip(Kii_list[(ind2-1):0:-1],range(1,ind2))]
        Kii_list[ind2][0] = -Kii_list[0][0]*sum(x*y for x,y in zip(tmp1,tmp2)) - Kii_list[0][0]*Hii_list[ind2][0]*Kii_list[0][ind2%2]
        tmp1 = [x[1] for x in Hii_list[1:ind2]]
        tmp2 = [x[(y+1)%2] for x,y in zip(Kii_list[(ind2-1):0:-1],range(1,ind2))]
        Kii_list[ind2][1] = -Kii_list[0][1]*sum(x*y for x,y in zip(tmp1,tmp2)) - Kii_list[0][1]*Hii_list[ind2][1]*Kii_list[0][(ind2+1)%2]

    part1 = [None,None]
    tmp1 = [x[0] for x in Hii_list[1:ind1]]
    tmp2 = [x[y%2] for x,y in zip(Kii_list[(ind1-1):0:-1],range(1,ind1))]
    part1[0] = (Delta_list[0][0]-x)*Kii_list[0][0]*sum(x*y for x,y in zip(tmp1,tmp2))
    tmp1 = [x[1] for x in Hii_list[1:ind1]]
    tmp2 = [x[(y+1)%2] for x,y in zip(Kii_list[(ind1-1):0:-1],range(1,ind1))]
    part1[1] = (Delta_list[0][1]-x)*Kii_list[0][1]*sum(x*y for x,y in zip(tmp1,tmp2))

    return np.stack([part1[0], part1[1]], axis=0)
