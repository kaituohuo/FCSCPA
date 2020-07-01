import numpy as np

from FCSCPA import CPA_block_general, CPA_general, generate_hf_Bii
from FCSCPA import generate_binary_hf_average, generate_binary_hf_rand
from FCSCPA import generate_uniform_hf_average, generate_uniform_hf_rand
from FCSCPA import matA_to_Aii, Aii_to_matA


hfe = lambda x,y: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+1e-3))


def uniform_parameter(tag_complex=False, block_size=None):
    num_block = np.random.randint(5, 10)
    if block_size is None:
        block_size = np.random.randint(1, 4)
    else: #used for compare with CPA_general
        assert isinstance(block_size, int) and block_size>0
    N1 = num_block * block_size
    matA = np.random.randn(N1,N1) + np.eye(N1)*N1
    matA = matA + matA.T
    if tag_complex:
        matA = matA + (np.random.randn(N1,N1)+np.eye(N1)*N1/2) * 1j
    x1 = np.random.rand()*2 - 1
    x2 = np.random.rand() + x1
    hf_rand = generate_uniform_hf_rand(x1, x2)
    hf_average = generate_uniform_hf_average(x1, x2)
    return num_block,block_size,matA,hf_rand,hf_average


def test_uniform_compare_with_CPA_general():
    num_block,block_size,matA,_,hf_average = uniform_parameter(block_size=1)
    matC_ = CPA_general(matA, hf_average)
    hf_Bii = generate_hf_Bii(np.ones((1,1)))
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average)[:,0,0]
    assert hfe(matC_, matC) < 1e-7, f'num_block={num_block}'


def test_uniform_real():
    num_block,block_size,matA,hf_rand,hf_average = uniform_parameter(tag_complex=False)
    hf_Bii = generate_hf_Bii(np.random.randn(block_size,block_size))
    N2 = 10000
    hf1 = lambda: Aii_to_matA(hf_Bii(hf_rand(num_block)).transpose(2,0,1))
    ret1 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average, tag_complex=False)
    ret3 = np.linalg.inv(matA - Aii_to_matA(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2<tmp1*2, f'num_block={num_block}, block_size={block_size}'


def test_uniform_complex():
    tag_complex = True
    num_block,block_size,matA,hf_rand,hf_average = uniform_parameter(tag_complex=True)
    hf_Bii = generate_hf_Bii(np.random.randn(block_size,block_size))
    N2 = 10000
    hf1 = lambda: Aii_to_matA(hf_Bii(hf_rand(num_block)).transpose(2,0,1))
    ret1 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average, tag_complex=True)
    ret3 = np.linalg.inv(matA - Aii_to_matA(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2, f'num_block={num_block}, block_size={block_size}'


def binary_parameter(tag_complex=False, block_size=None):
    num_block = np.random.randint(5, 10)
    if block_size is None:
        block_size = np.random.randint(1, 4)
    else: #used for compare with CPA_general
        assert isinstance(block_size, int) and block_size>0
    N1 = num_block * block_size
    matA = np.random.randn(N1,N1) + np.eye(N1)*N1
    if tag_complex:
        matA = matA + (np.random.randn(N1,N1)+np.eye(N1)*N1/2) * 1j
    x1 = np.random.rand()
    x2 = np.random.rand() + x1
    c1 = np.random.rand()
    hf_rand = generate_binary_hf_rand(x1, c1, x2)
    hf_average = generate_binary_hf_average(x1, c1, x2)
    return num_block,block_size,matA,hf_rand,hf_average


def test_binary_compare_with_CPA_general():
    num_block,block_size,matA,_,hf_average = binary_parameter(block_size=1)
    matC_ = CPA_general(matA, hf_average)
    hf_Bii = generate_hf_Bii(np.ones((1,1)))
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average)[:,0,0]
    tmp1 = hfe(matC_, matC)
    assert tmp1 < 1e-7, f'num_block={num_block}'


def test_binary_real():
    num_block,block_size,matA,hf_rand,hf_average = binary_parameter(tag_complex=False)
    hf_Bii = generate_hf_Bii(np.random.randn(block_size,block_size))
    N2 = 10000
    hf1 = lambda: Aii_to_matA(hf_Bii(hf_rand(num_block)).transpose(2,0,1))
    ret1 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average, tag_complex=False)
    ret3 = np.linalg.inv(matA - Aii_to_matA(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2, f'num_block={num_block}, block_size={block_size}'


def test_binary_complex():
    tag_complex = True
    num_block,block_size,matA,hf_rand,hf_average = binary_parameter(tag_complex=True)
    hf_Bii = generate_hf_Bii(np.random.randn(block_size,block_size))
    N2 = 10000
    hf1 = lambda: Aii_to_matA(hf_Bii(hf_rand(num_block)).transpose(2,0,1))
    ret1 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    ret2 = sum(np.linalg.inv(matA - hf1()) for _ in range(N2)) / N2
    matC = CPA_block_general(matA, block_size, hf_Bii, hf_average, tag_complex=True)
    ret3 = np.linalg.inv(matA - Aii_to_matA(matC))
    tmp1 = hfe(ret1,ret2)
    tmp2 = hfe((ret1+ret2)/2, ret3)
    assert tmp2 < tmp1*2, f'num_block={num_block}, block_size={block_size}'
