import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from RGF import inv_tridiagonal_matrix, build_device_Hamiltonian_with_ham0_ham1
from FCSCPA import FCS_CPA_oneband, FCS_CPA_multiband
from FCSCPA import generate_binary_hf_rand, generate_binary_hf_average
from FCSCPA import generate_uniform_hf_rand, generate_uniform_hf_average
from FCSCPA import generate_hf_Bii, Aii_to_matA


hfH = lambda x: np.conjugate(x.T)


# NAPL: Number of Atom Per Layer
# NBPA: Number of Band Per Atom
# NBPL: Number of Band Per Atom
def quick_transmission_moment_BF(energy_i_epsj, HInfo, num_order, num_sample, hf_rand, hf_Bii=None):
    NBPL = HInfo['ham_left_intra'].shape[0]
    NBPA = 1 if (hf_Bii is None) else hf_Bii(0).shape[0]
    energyEye1 = energy_i_epsj*np.eye(HInfo['ham_left_intra'].shape[0])

    tmp1 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], energyEye1-HInfo['ham_left_intra'], -hfH(HInfo['ham_left_inter']))
    self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_left_part']))
    line_width_left = 1j * (self_energy_left - hfH(self_energy_left))

    tmp1 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], energyEye1-HInfo['ham_right_intra'], -hfH(HInfo['ham_right_inter']))
    self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_right_part']))
    line_width_right = 1j * (self_energy_right - hfH(self_energy_right))

    ret = []
    inv_Green_part = energy_i_epsj*np.eye(HInfo['ham_central'].shape[0]) - HInfo['ham_central']
    inv_Green_part[:NBPL,:NBPL] -= self_energy_left
    inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right
    for _ in range(num_sample):
        if hf_Bii is None:
            inv_Green = inv_Green_part - np.diag(hf_rand(inv_Green_part.shape[0]))
        else:
            inv_Green = inv_Green_part - Aii_to_matA(hf_Bii(hf_rand(inv_Green_part.shape[0]//NBPA)).transpose(2,0,1))
        tmp0 = np.linalg.inv(inv_Green)[(-NBPL):,:NBPL]
        T = tmp0 @ line_width_left @ hfH(tmp0) @ line_width_right
        T_i = T
        moment_i = [T_i.trace()]
        for _ in range(num_order-1):
            T_i = np.matmul(T_i, T)
            moment_i.append(T_i.trace())
        assert max(abs(x.imag) for x in moment_i) < 1e-5
        ret.append([x.real for x in moment_i])
    ret = np.array(ret).T
    return ret


def quick_transmission_moment_FCSCPA(energy_i_epsj, HInfo, num_order, hf_average, hf_Bii=None):
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
    if hf_Bii is None:
        N_Gamma_trace = FCS_CPA_oneband(num_order, inv_Green_part, -line_width_right, line_width_left, hf_average)
    else:
        N_Gamma_trace = FCS_CPA_multiband(num_order, inv_Green_part, -line_width_right, line_width_left, hf_Bii, hf_average)
    ret = np.array([(1-2*(ind1%2))*x/2 for ind1,x in enumerate(N_Gamma_trace)])
    tmp0 = np.abs(ret.imag).max()
    if tmp0>1e-5:
        print('[warning] large image part "{}j" for energy={}'.format(tmp0, energy_i_epsj))
    return ret

# TODO multiprocessing
def demo_square_lattice_oneband_binary():
    # parameter
    num_layer = 2
    NAPL = 4 #Number of Atoms Per Layer
    num_BF_sample = 100
    num_moment = 4

    EFNN = -1 #Energy of First Nearest Neighbor
    EOnsite_central = 0
    EOnsite_left = 0
    EOnsite_right = 0
    EDisorder = 0.2
    hf_rand = generate_binary_hf_rand(-EDisorder, 0.5, EDisorder)
    hf_average = generate_binary_hf_average(-EDisorder, 0.5, EDisorder)
    energy = np.linspace(-4, 4, 50)
    energy_epsj = 1e-7j

    # calculation
    ham0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    ham1 = EFNN * np.eye(NAPL)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_square_lattice_multiband_binary():
    ## parameter
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
    hf_rand = generate_binary_hf_rand(-EDisorder, 0.5, EDisorder)
    hf_average = generate_binary_hf_average(-EDisorder, 0.5, EDisorder)
    hf_Bii = generate_hf_Bii(np.array([[0,1],[1,0]]))
    energy = np.linspace(-4, 4, 50)
    energy_epsj = 1e-7j

    # calculation
    ham0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    ham0 = np.kron(ham0, np.eye(NBPA))
    ham1 = EFNN * np.eye(NBPA*NAPL)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand, hf_Bii) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average, hf_Bii) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4 #fail sometimes
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_zigzag_lattice_oneband_binary():
    # parameter
    num_layer = 2
    NAPL = 4 #Number of Atoms Per Layer
    num_BF_sample = 100
    num_moment = 4

    EFNN = -1 #Energy of First Nearest Neighbor
    EOnsite_central = 0
    EOnsite_left = 0
    EOnsite_right = 0
    EDisorder = 0.2
    hf_rand = generate_binary_hf_rand(-EDisorder, 0.5, EDisorder)
    hf_average = generate_binary_hf_average(-EDisorder, 0.5, EDisorder)
    energy = np.linspace(-3, 3, 50)
    energy_epsj = 1e-7j

    # calculation
    assert NAPL%4==0
    ham0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    tmp1 = EFNN * np.kron(np.eye(NAPL//4+1), np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]))
    ham1 = tmp1[:NAPL, :NAPL]
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_zigzag_lattice_multiband_binary():
    ## parameter
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
    hf_rand = generate_binary_hf_rand(-EDisorder, 0.5, EDisorder)
    hf_average = generate_binary_hf_average(-EDisorder, 0.5, EDisorder)
    hf_Bii = generate_hf_Bii(np.array([[0,1],[1,0]]))
    energy = np.linspace(-4, 4, 50)
    energy_epsj = 1e-7j

    # calculation
    assert NAPL%4==0
    tmp0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    ham0 = np.kron(tmp0, np.eye(2))
    tmp1 = EFNN * np.kron(np.eye(NAPL//4+1), np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]))
    ham1 = np.kron(tmp1[:NAPL, :NAPL], np.eye(2))
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand, hf_Bii) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average, hf_Bii) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4 #fail
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_zigzag_lattice_oneband_uniform():
    ## parameter
    num_layer = 2
    NAPL = 4 #Number of Atoms Per Layer
    NBPA = 2 #Number of Bands Per Atom
    num_BF_sample = 1000
    num_moment = 4

    EFNN = -1 #Energy of First Nearest Neighbor
    EOnsite_central = 0
    EOnsite_left = 0
    EOnsite_right = 0
    EDisorder = 0.2
    hf_rand = generate_uniform_hf_rand(-EDisorder, EDisorder)
    hf_average = generate_uniform_hf_average(-EDisorder, EDisorder)
    energy = np.linspace(-4, 4, 50)
    energy_epsj = 1e-7j

    # calculation
    assert NAPL%4==0
    ham0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    ham1 = EFNN * np.kron(np.eye(NAPL//4+1), np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]))[:NAPL, :NAPL]
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4 #fail
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_zigzag_lattice_multiband_uniform():
    # TODO fail, hf_average(multiband) cannot solve
    ## parameter
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
    energy = np.linspace(-4, 4, 50)
    energy_epsj = 1e-7j

    # calculation
    assert NAPL%4==0
    tmp0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
    ham0 = np.kron(tmp0, np.eye(2))
    tmp1 = EFNN * np.kron(np.eye(NAPL//4+1), np.array([[0,0,0,0],[1,0,0,0],[0,0,0,1],[0,0,0,0]]))
    ham1 = np.kron(tmp1[:NAPL, :NAPL], np.eye(2))
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    tmp0 = [quick_transmission_moment_BF(x+energy_epsj, HInfo, num_moment, num_BF_sample, hf_rand, hf_Bii) for x in tqdm(energy)]
    T_moment_BF = np.stack(tmp0, axis=1)

    tmp0 = [quick_transmission_moment_FCSCPA(x+energy_epsj, HInfo, num_moment, hf_average, hf_Bii) for x in tqdm(energy)]
    T_moment_FCSCPA = np.stack(tmp0, axis=1)

    ## figure
    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4 #fail
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(energy, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(energy, y, 'x', color=z, markersize=2)
    ax.legend()
