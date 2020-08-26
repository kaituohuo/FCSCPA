import itertools
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from RGF import inv_tridiagonal_matrix, build_device_Hamiltonian_with_ham0_ham1
from RGF.phonon_utils import (hf_wave_numer_to_angular_frequency,
        FORCE_CONSTANT_Dresselhaus, quick_phonon_transmission, hf_dynamic_matrix, build_phonon_ham0)
from FCSCPA import FCS_CPA_oneband, FCS_CPA_multiband
from FCSCPA import generate_binary_hf_rand, generate_binary_hf_average
from FCSCPA import generate_uniform_hf_rand, generate_uniform_hf_average
from FCSCPA import generate_hf_Bii, Aii_to_matA

# TODO check RGF.phonon_utils.FORCE_CONSTANT_Dresselhaus units
# TODO check force_constant_plus_minus_sign

hfH = lambda x: np.conjugate(x.T)
hfe = lambda x,y,eps=1e-5: np.max(np.abs(x-y)/(np.abs(x)+np.abs(y)+eps))


def quick_phonon_transmission_moment_BF(angular_frequency_i, mass_left_lead,
            mass_right_lead, HInfo, num_order, num_sample, hf_rand, hf_Bii, energy_epsj):
    # hf_Bii cannot be None for phonon, since mass_central is included in hf_Bii
    NBPL = HInfo['ham_left_intra'].shape[0]
    NBPA = hf_Bii(0).shape[0]
    NAPL = NBPL // NBPA
    num_layer = HInfo['ham_central'].shape[0] // NBPL

    tmp1 = (mass_left_lead*angular_frequency_i**2 + energy_epsj) * np.eye(HInfo['ham_left_intra'].shape[0]) - HInfo['ham_left_intra']
    tmp2 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], tmp1, -hfH(HInfo['ham_left_inter']))
    self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_left_part']))
    line_width_left = 1j * (self_energy_left - hfH(self_energy_left))

    tmp1 = (mass_right_lead*angular_frequency_i**2 + energy_epsj) * np.eye(HInfo['ham_right_intra'].shape[0]) - HInfo['ham_right_intra']
    tmp2 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], tmp1, -hfH(HInfo['ham_right_inter']))
    self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_right_part']))
    line_width_right = 1j * (self_energy_right - hfH(self_energy_right))

    ret = []
    inv_Green_part = np.eye(HInfo['ham_central'].shape[0])*energy_epsj - HInfo['ham_central']
    inv_Green_part[:NBPL,:NBPL] -= self_energy_left
    inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right
    for _ in range(num_sample):
        inv_Green = inv_Green_part - Aii_to_matA(hf_Bii(hf_rand(NAPL*num_layer)).transpose(2,0,1))
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


def quick_phonon_transmission_moment_FCSCPA(angular_frequency_i, mass_left_lead,
            mass_right_lead, HInfo, num_order, hf_average, hf_Bii, energy_epsj, matC0=None):
    NBPL = HInfo['ham_left_intra'].shape[0]
    NBPA = hf_Bii(0).shape[0]
    NAPL = NBPL // NBPA
    num_layer = HInfo['ham_central'].shape[0] // NBPL

    tmp1 = (mass_left_lead*angular_frequency_i**2 + energy_epsj) * np.eye(HInfo['ham_left_intra'].shape[0]) - HInfo['ham_left_intra']
    tmp2 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], tmp1, -hfH(HInfo['ham_left_inter']))
    self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_left_part']))
    line_width_left = 1j * (self_energy_left - hfH(self_energy_left))
    line_width_left = np.pad(line_width_left, [(0,NBPL*(num_layer-1)), (0,NBPL*(num_layer-1))], mode='constant')

    tmp1 = (mass_right_lead*angular_frequency_i**2 + energy_epsj) * np.eye(HInfo['ham_right_intra'].shape[0]) - HInfo['ham_right_intra']
    tmp2 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], tmp1, -hfH(HInfo['ham_right_inter']))
    self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_right_part']))
    line_width_right = 1j * (self_energy_right - hfH(self_energy_right))
    line_width_right = np.pad(line_width_right, [(NBPL*(num_layer-1),0), (NBPL*(num_layer-1),0)], mode='constant')

    inv_Green_part = np.eye(HInfo['ham_central'].shape[0])*energy_epsj - HInfo['ham_central']
    inv_Green_part[:NBPL,:NBPL] -= self_energy_left
    inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right

    N_Gamma_trace,matC0 = FCS_CPA_multiband(num_order, inv_Green_part,
            line_width_right, line_width_left, hf_Bii, hf_average, matC0=matC0)
    ret = -1/2 * N_Gamma_trace
    if np.abs(ret.imag).max()>1e-5:
        print('[warning] large image part "{}j" for energy={}'.format(np.abs(ret.imag).max(), angular_frequency_i))
    return ret, matC0


def demo_square_lattice_oneband_binary():
    # parameter
    num_layer = 20
    NAPL = 20 #Number of Atom Per Layer
    num_sample = 1000
    num_moment = 4
    wave_number = np.linspace(10, 1700, 100) #cm-1
    angular_frequency = hf_wave_numer_to_angular_frequency(wave_number) #equivalent to Hz
    energy_epsj = 1e-6j
    mass_carbon = 12.0107 #in AMU
    mass_left_lead = mass_carbon
    mass_right_lead = mass_carbon
    hf_rand = generate_binary_hf_rand(0.9*mass_carbon, 0.5, 1.1*mass_carbon)
    hf_average = generate_binary_hf_average(0.9*mass_carbon, 0.5, 1.1*mass_carbon)

    # Hamiltonian
    force_constant = -FORCE_CONSTANT_Dresselhaus[0,0] #see RGF.phonon_utils.FORCE_CONSTANT_Dresselhaus
    ham1 = np.ones((1,1))*force_constant
    ham0 = np.ones((1,1)) * (-2*force_constant)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    T_moment_BF = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        T_moment_BF.append(quick_phonon_transmission_moment_BF(x, mass_left_lead,
                mass_right_lead, HInfo, num_moment, num_sample, hf_rand, hf_Bii, energy_epsj))
    T_moment_BF = np.stack(T_moment_BF, axis=1)

    matC0 = None
    T_moment_FCSCPA = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        tmp0,matC0 = quick_phonon_transmission_moment_FCSCPA(x, mass_left_lead,
                mass_right_lead, HInfo, num_moment, hf_average, hf_Bii, energy_epsj, matC0=matC0)
        T_moment_FCSCPA.append(tmp0)
    T_moment_FCSCPA = np.stack(T_moment_FCSCPA, axis=1)
    # use previous matC0 doesn't make any help

    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(wave_number, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(wave_number, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_square_lattice_multiband_binary():
    # parameter
    num_layer = 2
    NAPL = 4 #Number of Atom Per Layer
    num_sample = 1000
    num_moment = 4
    wave_number = np.linspace(10, 1780, 100) #cm-1
    angular_frequency = hf_wave_numer_to_angular_frequency(wave_number) #equivalent to Hz
    energy_epsj = 1e-6j
    mass_carbon = 12.0107 #in AMU
    mass_left_lead = mass_carbon
    mass_right_lead = mass_carbon
    hf_rand = generate_binary_hf_rand(0.9*mass_carbon, 0.5, 1.1*mass_carbon)
    hf_average = generate_binary_hf_average(0.9*mass_carbon, 0.5, 1.1*mass_carbon)

    # Hamiltonian
    ham1 = np.kron(np.eye(NAPL), hf_dynamic_matrix(30))
    if NAPL>1:
        tmp1 = np.kron(np.diag(np.ones(NAPL-1),1), hf_dynamic_matrix(120))
        ham0 = tmp1 + tmp1.T
    else:
        ham0 = np.zeros(3)
    ham0 = build_phonon_ham0(3, ham0, ham1)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    T_moment_BF = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        T_moment_BF.append(quick_phonon_transmission_moment_BF(x, mass_left_lead,
                mass_right_lead, HInfo, num_moment, num_sample, hf_rand, hf_Bii, energy_epsj))
    T_moment_BF = np.stack(T_moment_BF, axis=1)

    matC0 = None
    T_moment_FCSCPA = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        tmp0, matC0 = quick_phonon_transmission_moment_FCSCPA(x,
                mass_left_lead, mass_right_lead, HInfo, num_moment, hf_average, hf_Bii, energy_epsj, matC0=matC0)
        T_moment_FCSCPA.append(tmp0)
    T_moment_FCSCPA = np.stack(T_moment_FCSCPA, axis=1)

    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(wave_number, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(wave_number, y, 'x', color=z, markersize=2)
    ax.legend()


def demo_zigzag_lattice_multiband_binary():
    # parameter
    num_layer = 2
    NAPL = 4 #Number of Atom Per Layer
    num_sample = 1000
    num_moment = 4
    wave_number = np.linspace(10, 1600, 100) #cm-1
    angular_frequency = hf_wave_numer_to_angular_frequency(wave_number) #equivalent to Hz
    energy_epsj = 1e-6j
    mass_carbon = 12.0107 #in AMU
    mass_left_lead = mass_carbon
    mass_right_lead = mass_carbon
    hf_rand = generate_binary_hf_rand(0.9*mass_carbon, 0.5, 1.1*mass_carbon)
    hf_average = generate_binary_hf_average(0.9*mass_carbon, 0.5, 1.1*mass_carbon)

    # Hamiltonian
    hf1 = lambda x: slice(3*x,3*x+3)
    ham1 = np.zeros([3*NAPL,3*NAPL])
    for ind1,x in zip(range(NAPL),itertools.cycle([None,(-1,-30),(1,30),None])):
        if x is not None:
            ham1[hf1(ind1),hf1(ind1+x[0])] = hf_dynamic_matrix(x[1])
    ham0 = np.zeros([3*NAPL,3*NAPL])
    for ind1,theta in enumerate(np.tile(np.array([30,90,150,90]), [NAPL//4+1])[:(NAPL-1)]):
        ham0[hf1(ind1),hf1(ind1+1)] = hf_dynamic_matrix(theta)
    ham0 = build_phonon_ham0(3, ham0+ham0.T, ham1)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    T_moment_BF = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        T_moment_BF.append(quick_phonon_transmission_moment_BF(x, mass_left_lead,
                mass_right_lead, HInfo, num_moment, num_sample, hf_rand, hf_Bii, energy_epsj))
    T_moment_BF = np.stack(T_moment_BF, axis=1)

    matC0 = None
    T_moment_FCSCPA = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        tmp0,matC0 = quick_phonon_transmission_moment_FCSCPA(x,
                mass_left_lead, mass_right_lead, HInfo, num_moment, hf_average, hf_Bii, energy_epsj, matC0=matC0)
        T_moment_FCSCPA.append(tmp0)
    T_moment_FCSCPA = np.stack(T_moment_FCSCPA, axis=1)

    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(wave_number, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(wave_number, y, 'x', color=z, markersize=2)
    ax.legend()
    # ax.set_ylim(-0.1, 5) #always diverge for large frequency


def demo_square_lattice_multiband_uniform():
    # parameter
    num_layer = 2
    NAPL = 4 #Number of Atom Per Layer
    num_sample = 1000
    num_moment = 4
    wave_number = np.linspace(10, 900, 100) #cm-1
    angular_frequency = hf_wave_numer_to_angular_frequency(wave_number) #equivalent to Hz
    energy_epsj = 1e-6j
    mass_carbon = 12.0107 #in AMU
    mass_left_lead = mass_carbon
    mass_right_lead = mass_carbon
    hf_rand = generate_uniform_hf_rand(0.9*mass_carbon, 1.1*mass_carbon)
    hf_average = generate_uniform_hf_average(0.9*mass_carbon, 1.1*mass_carbon)

    # Hamiltonian
    ham1 = np.kron(np.eye(NAPL), hf_dynamic_matrix(30))
    if NAPL>1:
        tmp1 = np.kron(np.diag(np.ones(NAPL-1),1), hf_dynamic_matrix(120))
        ham0 = tmp1 + tmp1.T
    else:
        ham0 = np.zeros(3)
    ham0 = build_phonon_ham0(3, ham0, ham1)
    HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

    T_moment_BF = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        T_moment_BF.append(quick_phonon_transmission_moment_BF(x, mass_left_lead,
                mass_right_lead, HInfo, num_moment, num_sample, hf_rand, hf_Bii, energy_epsj))
    T_moment_BF = np.stack(T_moment_BF, axis=1)

    matC0 = None
    T_moment_FCSCPA = []
    for x in tqdm(angular_frequency):
        hf_Bii = generate_hf_Bii(-np.ones((1,1))*x**2)
        tmp0,matC0 = quick_phonon_transmission_moment_FCSCPA(x,
                mass_left_lead, mass_right_lead, HInfo, num_moment, hf_average, hf_Bii, energy_epsj, matC0=matC0)
        T_moment_FCSCPA.append(tmp0)
    T_moment_FCSCPA = np.stack(T_moment_FCSCPA, axis=1)

    # assert np.abs(T_moment_FCSCPA.imag).max() < 1e-4
    tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
    fig,ax = plt.subplots()
    for x,y,z in zip(range(num_moment), T_moment_FCSCPA.real, tableau_colorblind):
        ax.plot(wave_number, y, color=z, label='$tr(T^{}$)'.format(x+1))
    for x,y,z in zip(range(num_moment), T_moment_BF.real.mean(axis=2), tableau_colorblind):
        ax.plot(wave_number, y, 'x', color=z, markersize=2)
    ax.legend()
