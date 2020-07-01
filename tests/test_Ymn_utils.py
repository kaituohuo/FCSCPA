import random
import numpy as np

from FCSCPA import ball_in_bowl, Ymn_Leibniz_matAB, demo_Ymn_Leibniz, demo_Ymn_Binomial, Ymn_electron_zeroK

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def test_ball_in_bowl():
    def hf1(x0, x1):
        ret0 = []
        ret1 = []
        for y1,y2 in ball_in_bowl(x0, x1):
            ret0.append(y1)
            ret1.append(y2)
        return ret0, ret1
    tmp1,tmp2 = hf1(0, 3)
    assert [] == tmp1
    assert [] == tmp2

    tmp1,tmp2 = hf1(3, 1)
    assert [(3,)] == tmp1
    assert [(0,0,0,1)] == tmp2

    tmp1,tmp2 = hf1(3, 2)
    assert [(0,3),(1,2),(2,1),(3,0)] == tmp1
    assert [(1,0,0,1),(0,1,1,0),(0,1,1,0),(1,0,0,1)] == tmp2


def test_Ymn_Leibniz_matAB():
    tmp1,tmp2 = Ymn_Leibniz_matAB(1, 1)
    assert hfe(np.array([1]), tmp1) < 1e-7
    assert [[1,0]] == tmp2.tolist()

    tmp1,tmp2 = Ymn_Leibniz_matAB(1, 2)
    assert hfe(np.array([1]), tmp1) < 1e-7
    assert [[0,1]] == tmp2.tolist()

    tmp1,tmp2 = Ymn_Leibniz_matAB(2, 4)
    assert hfe(np.array([6,8]), tmp1) < 1e-7
    assert [[0,2],[2,0]] == tmp2.tolist()


def test_Ymn():
    fL = random.uniform(0, 1)
    fR = random.uniform(0, 1)
    for m in range(1, 7):
        for n in range(1, 7):
            tmp1 = demo_Ymn_Leibniz(m, n, fL, fR, 'electron')
            tmp2 = demo_Ymn_Binomial(m, n, fL, fR, 'electron')
            assert abs(tmp1-tmp2) < 1e-7
            tmp1 = demo_Ymn_Leibniz(m, n, fL, fR, 'phonon')
            tmp2 = demo_Ymn_Binomial(m, n, fL, fR, 'phonon')
            assert abs(tmp1-tmp2) < 1e-7


def test_Ymn_electron_zeroK():
    ground_truth = [
        (1, [1]),
        (2, [1,2]),
        (3, [1,6,6]),
        (4, [1,14,36,24]),
        (5, [1,30,150,240,120]),
    ]
    for x,ret_ in ground_truth:
        ret = Ymn_electron_zeroK(x)
        assert np.all(np.array(ret_)==ret)
