import numpy as np

from FCSCPA import CPA_general
from FCSCPA import generate_binary_hf_average, generate_binary_hf_rand
from FCSCPA import generate_uniform_hf_average, generate_uniform_hf_rand

hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def uniform_parameter(tag_complex=False):
    N1 = np.random.randint(5, 10)
    matA = np.random.randn(N1,N1) + np.eye(N1)*N1/2
    if tag_complex:
        matA = matA + (np.random.randn(N1,N1)+np.eye(N1)*N1/2) * 1j
    x1 = np.random.rand()*2 - 1
    x2 = np.random.rand() + x1
    hf_rand = generate_uniform_hf_rand(x1, x2)
    hf_average = generate_uniform_hf_average(x1, x2)
    return N1,matA,hf_rand,hf_average


def test_real_value_uniform():
    N1,matA,hf_rand,hf_average = uniform_parameter(tag_complex=False)
    N2 = 10000
    ret1 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    matC = CPA_general(matA, hf_average, tag_complex=False)
    ret3 = np.linalg.inv(matA-np.diag(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2


def test_complex_value_uniform():
    N1,matA,hf_rand,hf_average = uniform_parameter(tag_complex=True)
    N2 = 10000
    ret1 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    matC = CPA_general(matA, hf_average, tag_complex=True)
    ret3 = np.linalg.inv(matA-np.diag(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2


def binary_parameter(tag_complex=False):
    N1 = np.random.randint(5, 10)
    matA = np.random.randn(N1,N1) + np.eye(N1)*N1
    if tag_complex:
        matA = matA + (np.random.randn(N1,N1)+np.eye(N1)*N1/2) * 1j
    x1 = np.random.rand()
    x2 = np.random.rand() + x1
    c1 = np.random.rand()
    hf_rand = generate_binary_hf_rand(x1, c1, x2)
    hf_average = generate_binary_hf_average(x1, c1, x2)
    return N1,matA,hf_rand,hf_average


def test_real_value_binary():
    N1,matA,hf_rand,hf_average = binary_parameter(tag_complex=False)
    N2 = 10000
    ret1 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    matC = CPA_general(matA, hf_average, tag_complex=False)
    ret3 = np.linalg.inv(matA-np.diag(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2


def test_complex_value_binary():
    N1,matA,hf_rand,hf_average = binary_parameter(tag_complex=True)
    N2 = 10000
    ret1 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - np.diag(hf_rand(N1))) for _ in range(N2)) / N2
    matC = CPA_general(matA, hf_average, tag_complex=True)
    ret3 = np.linalg.inv(matA-np.diag(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2
