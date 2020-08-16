import numpy as np
import quadpy

from RGF import inv_tridiagonal_matrix, build_device_Hamiltonian_with_ham0_ham1
from FCSCPA import FCS_CPA_multiband, CPA_block_general, generate_uniform_hf_rand, generate_hf_Bii, Aii_to_matA

hfH = lambda x: np.conjugate(x.T)

def generate_uniform_hf_average(x1, x2):
    assert x2 > x1
    tag = hasattr(quadpy, 'line_segment')
    def hf_average(hf1):
        if tag:
            ret = quadpy.line_segment.integrate_adaptive(hf1, [x1,x2], 1e-5)[0]/(x2-x1)
        else:
            ret = quadpy.c1.integrate_adaptive(hf1, [x1,x2], 1e-5)[0]/(x2-x1)
        return ret
    return hf_average


energy = 2.04

num_layer = 2
NAPL = 4 #Number of Atoms Per Layer
NBPA = 2 #Number of Bands Per Atom
num_BF_sample = 100
num_moment = 4

EFNN = -1 #Energy of First Nearest Neighbor
EOnsite_central = 0
EOnsite_left = 0
EOnsite_right = 0
EDisorder = 0.2
hf_rand = generate_uniform_hf_rand(-EDisorder, EDisorder)
hf_average = generate_uniform_hf_average(-EDisorder, EDisorder)
hf_Bii = generate_hf_Bii(np.array([[0,1],[1,0]]))
energy_i_epsj = 2.04 + 1e-6j

assert NAPL%4==0
tmp0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
ham0 = np.kron(tmp0, np.eye(2))
tmp1 = EFNN * np.kron(np.eye(NAPL//4+1), np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]))
ham1 = np.kron(tmp1[:NAPL, :NAPL], np.eye(2))
HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

NBPL = HInfo['ham_left_intra'].shape[0]
num_layer = HInfo['ham_central'].shape[0] // NBPL
energyEye1 = energy_i_epsj*np.eye(HInfo['ham_left_intra'].shape[0])

tmp1 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], energyEye1-HInfo['ham_left_intra'], -hfH(HInfo['ham_left_inter']))
self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_left_part']))
line_width_left = 1j * (self_energy_left - hfH(self_energy_left))
line_width_left = np.pad(line_width_left, [(0,NBPL*(num_layer-1)), (0,NBPL*(num_layer-1))], mode='constant')

tmp1 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], energyEye1-HInfo['ham_right_intra'], -hfH(HInfo['ham_right_inter']))
self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_right_part']))
line_width_right = 1j * (self_energy_right - hfH(self_energy_right))
line_width_right = np.pad(line_width_right, [(NBPL*(num_layer-1),0), (NBPL*(num_layer-1),0)], mode='constant')

inv_Green_part = energy_i_epsj*np.eye(HInfo['ham_central'].shape[0]) - HInfo['ham_central']
inv_Green_part[:NBPL,:NBPL] -= self_energy_left
inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right


NBPA = hf_Bii(0).shape[0]
assert inv_Green_part.shape[0]%NBPA==0
num_atom = inv_Green_part.shape[0]//NBPA
N_list = [[None,None] for _ in range(2*num_moment)]
Nii_list = [[None,None] for _ in range(2*num_moment)]
Delta_list = [[None,None] for _ in range(2*num_moment)]

delta00 = CPA_block_general(inv_Green_part, NBPA, hf_Bii, hf_average)

N_Gamma_trace = FCS_CPA_multiband(num_moment, inv_Green_part, -line_width_right, line_width_left, hf_Bii, hf_average)
ret = np.array([(1-2*(ind1%2))*x/2 for ind1,x in enumerate(N_Gamma_trace)])

print(ret)
[1.96562606, 1.93251924, 1.90062872, 1.86990601]
