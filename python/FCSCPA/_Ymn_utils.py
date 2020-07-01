import math
import numpy as np
import scipy.linalg
import scipy.special
from functools import reduce
from collections import Counter
from itertools import combinations

def ball_in_bowl(num_ball:int, num_bowl:int):
    '''
    put N0(num_ball) balls into N1(num_bowl) bowls

    (ret1)(yield,(tuple,int,N1),N2): number of balls in each bowl
    (ret2)(yield,(tuple,int,N0+1),N2): number of bowls with x balls (from 0 to N0)
    '''
    if num_ball==0:
        yield from []
    elif num_bowl==1:
        yield (num_ball,), (0,)*num_ball+(1,)
    else:
        assert num_ball>0 and num_bowl>1
        for x in combinations(list(range(1,num_ball+num_bowl)), num_bowl-1):
            tmp1 = tuple(y1-y0-1 for y0,y1 in zip((0,)+x, x+(num_ball+num_bowl,)))
            tmp2 = Counter(tmp1)
            yield tmp1, tuple(tmp2.get(y,0) for y in range(0,num_ball+1))


def Ymn_Leibniz_matAB(m:int, n:int)->(np.ndarray, np.ndarray):
    '''n-th order derivative for Y to m-th order expression

    m(int): exponent
    n(int): n-th order derivative
    (ret0)(np,float,(N0,)): matA
    (ret1)(np.int,(N0,2)): matB
    '''
    assert m <= n
    if m==n:
        matA = np.array([math.factorial(n)])
        matB = np.array([[n,0]], dtype=np.int)
    else:
        all_case = dict()
        for _,x in ball_in_bowl(n-m, m):
            tmp1 = [math.factorial(y0+1)**y1 for y0,y1 in enumerate(x)]
            tmp1 = math.factorial(n) / reduce(lambda x,y:x*y, tmp1)
            tmp2 = (sum(x[::2]),sum(x[1::2]))
            all_case[tmp2] = all_case.get(tmp2,0) + tmp1
        tmp1 = sorted(all_case.items(), key=lambda x:x[0])
        matA = np.array([x[1] for x in tmp1])
        matB = np.array([x[0] for x in tmp1], dtype=np.int)
    return matA, matB



def Ymn_Leibniz(fL:float, fR:float, matA:np.ndarray, matB:np.ndarray, tag_type:str='electron')->float:
    '''
    see demo_Ymn_Leibniz for basic usage
    generate matA and matB using Ymn_Leibniz_matAB
    '''
    assert tag_type in {'electron', 'phonon'}
    assert matA.ndim==1 and matB.ndim==2 and matA.shape[0]==matB.shape[0]
    if tag_type == 'electron':
        odd_taylor = fL - fR
        even_taylor = fL + fR - 2*fL*fR
    else:
        odd_taylor = fL - fR
        even_taylor = fL + fR + 2*fL*fR
    ret = np.sum(matA * (odd_taylor**matB[:,0] * even_taylor**matB[:,1])).item()
    return ret


def demo_Ymn_Leibniz(m:int, n:int, fL:float, fR:float, tag_type:str='electron')->float:
    '''
    demo how to calculate Ymn using General Leibniz rule
    (h x omega) will be ignored for phonon
    '''
    if m>n:
        return 0
    matA,matB = Ymn_Leibniz_matAB(m, n)
    ret = Ymn_Leibniz(fL, fR, matA, matB, tag_type)
    return ret


def demo_Ymn_Binomial(m:int, n:int, fL:float, fR:float, tag_type:str='electron')->float:
    '''
    demo how to calculate Ymn using Binomial coefficient
    (h x omega) will be ignored for phonon
    '''
    assert tag_type in {'electron', 'phonon'}
    ret = 0
    tmp0 = scipy.special.comb(m, np.arange(0,m+1))
    if tag_type == 'electron':
        x1,x2,x3,x4 = fL, 1-fR, fR, 1-fL
    else:
        x1,x2,x3,x4 = fL, 1+fR, fR, 1+fL
    for r,tmp0 in enumerate(scipy.special.comb(m, np.arange(0,m+1))):
        s = np.arange(0, r+1)[:,np.newaxis]
        t = np.arange(0, m-r+1)[np.newaxis]
        tmp1 = np.array([[0 if x==y else (x-y)**n for y in t[0]] for x in s[:,0]])
        tmp2 = scipy.special.comb(r,s) * scipy.special.comb(m-r,t) * tmp1 * (-1)**(s+t)
        ret += tmp0 * x1**r * x2**r * x3**(m-r) * x4**(m-r) * tmp2.sum()
    return (-1)**m * ret


def Ymn_electron_zeroK(n:int)->np.ndarray:
    '''
    to retrieve T-coefficient: [(-1)**x*y/(x+1) for x,y in enumerate(Ymn_electron_zeroK(2))]
    (ret)(np,int,(n,))
    '''
    assert isinstance(n, int) and n>0
    ret = []
    for m in range(1,n+1):
        tmp1 = np.diag(scipy.linalg.pascal(m+1)[::-1])
        tmp2 = 1-(np.arange(m+1)%2)*2
        tmp3 = (m - np.arange(m+1))**n
        ret.append((tmp1*tmp2*tmp3).sum())
    return np.array(ret, dtype=np.int)
