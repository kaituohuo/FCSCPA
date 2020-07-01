import numpy as np
import scipy.constants
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from RGF import inv_tridiagonal_matrix, build_device_Hamiltonian_with_ham0_ham1
from RGF.phonon_utils import (HZ_FACTOR, hf_wave_numer_to_angular_frequency,
        FORCE_CONSTANT_Dresselhaus, quick_phonon_transmission, hf_dynamic_matrix, build_phonon_ham0)
from FCSCPA import FCS_CPA_oneband, FCS_CPA_multiband
from FCSCPA import generate_binary_hf_rand, generate_binary_hf_average
from FCSCPA import generate_hf_Bii, Aii_to_matA
from FCSCPA import Ymn_Leibniz_matAB, Ymn_Leibniz

hfH = lambda x: np.conjugate(x.T)

scipy.constants.physical_constants['speed of light in vacuum'] #scipy.constants.speed_of_light
scipy.constants.physical_constants['Planck constant'] #scipy.constants.Planck
scipy.constants.physical_constants['elementary charge'] #scipy.constants.elementary_charge
scipy.constants.physical_constants['atomic mass constant']
scipy.constants.physical_constants['Boltzmann constant'] #scipy.constants.Boltzmann

def linear_factor_with_hw_one(temperature_left_kelvin, temperature_right_kelvin, wave_number_cm):
    assert temperature_left_kelvin > temperature_right_kelvin
    T = (temperature_left_kelvin + temperature_right_kelvin) / 2
    delta_T = temperature_left_kelvin - temperature_right_kelvin
    linear_factor = delta_T / (2 * T * kt_over_hbar_omega(T, wave_number_cm))
    return linear_factor


def kt_over_hbar_omega(temperature_Kelvin, wave_number_cm):
    '''the factor when treating hbar*omega as 1'''
    tmp1 = scipy.constants.Boltzmann * temperature_Kelvin
    tmp2 = scipy.constants.Planck * wave_number_cm * 100 * scipy.constants.speed_of_light
    return tmp1 / tmp2


def generate_distribution_function(temperature_kelvin):
    tmp1 = scipy.constants.Planck / (2*np.pi*HZ_FACTOR*scipy.constants.Boltzmann*temperature_kelvin)
    def hf_distribution_function(angular_frequency):
        return 1./(np.exp(angular_frequency*tmp1)-1)
    return hf_distribution_function


## parameter
num_layer = 2
NAPL = 4 #Number of Atom Per Layer
num_cumulant = 8
wave_number = 600 #cm-1
angular_frequency = hf_wave_numer_to_angular_frequency(wave_number) #equivalent to Hz
energy_epsj = 1e-7j

mass_carbon = 12.0107 #in AMU
mass_left_lead = mass_carbon
mass_right_lead = mass_carbon

# even (95,105) is far from small temperature limit
# when using T_L=98 and T_R=102, C_3 will be almost equal to C_1 for small Y_23 and cause a bad visual effect
temperature_left = 305 #Kelvin
hf_fL = generate_distribution_function(temperature_left)
fL = hf_fL(angular_frequency)
temperature_right = 295 #Kelvin
hf_fR = generate_distribution_function(temperature_right)
fR = hf_fR(angular_frequency)
linear_factor = linear_factor_with_hw_one(temperature_left, temperature_right, wave_number)

cumulant_coeff = []
for n in range(1, num_cumulant+1):
    tmp1 = []
    for m in range(1, n+1):
        matA,matB = Ymn_Leibniz_matAB(m, n)
        tmp1.append(Ymn_Leibniz(fL, fR, matA, matB, 'phonon') * (-1/(2*m)))
    cumulant_coeff.append(np.array(tmp1))

hf_Bii = generate_hf_Bii(-angular_frequency**2*np.eye(3))
disorder_strength = np.linspace(0.01, 0.5) #large disorder will case large image part
mass_disorder = [((1-x)*mass_carbon, 0.5, (1+x)*mass_carbon) for x in disorder_strength]


## Hamiltonian
ham1 = np.kron(np.eye(NAPL), hf_dynamic_matrix(30))
if NAPL>1:
    tmp1 = np.kron(np.diag(np.ones(NAPL-1),1), hf_dynamic_matrix(120))
    ham0 = tmp1 + tmp1.T
else:
    ham0 = np.zeros(3)
ham0 = build_phonon_ham0(3, ham0, ham1)
HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)


## calculation
NBPL = hf_Bii(0).shape[0]*NAPL

tmp1 = (mass_left_lead*angular_frequency**2 + energy_epsj) * np.eye(HInfo['ham_left_intra'].shape[0]) - HInfo['ham_left_intra']
tmp2 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], tmp1, -hfH(HInfo['ham_left_inter']))
self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_left_part']))
line_width_left = 1j * (self_energy_left - hfH(self_energy_left))
line_width_left = np.pad(line_width_left, [(0,NBPL*(num_layer-1)), (0,NBPL*(num_layer-1))], mode='constant')

tmp1 = (mass_right_lead*angular_frequency**2 + energy_epsj) * np.eye(HInfo['ham_right_intra'].shape[0]) - HInfo['ham_right_intra']
tmp2 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], tmp1, -hfH(HInfo['ham_right_inter']))
self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp2, hfH(HInfo['ham_central_right_part']))
line_width_right = 1j * (self_energy_right - hfH(self_energy_right))
line_width_right = np.pad(line_width_right, [(NBPL*(num_layer-1),0), (NBPL*(num_layer-1),0)], mode='constant')

inv_Green_part = np.eye(HInfo['ham_central'].shape[0])*energy_epsj - HInfo['ham_central']
inv_Green_part[:NBPL,:NBPL] -= self_energy_left
inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right

cumulant = []
for mass_disorder_i in tqdm(mass_disorder):
    hf_average = generate_binary_hf_average(*mass_disorder_i)
    N_Gamma_trace = FCS_CPA_multiband(num_cumulant, inv_Green_part, line_width_right, line_width_left, hf_Bii, hf_average)
    tmp1 = np.array([np.dot(x, N_Gamma_trace[:x.shape[0]]) for x in cumulant_coeff])
    if np.abs(tmp1.imag).max()>1e-5:
        print('[WARNING] large image part "{}j" for mass_disorder_i={}'.format(np.abs(tmp1.imag).max(), mass_disorder_i))
    cumulant.append(tmp1)


## figure
fig,ax = plt.subplots(1,1)
tmp1 = np.array(cumulant).T.real
tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
for x1,x2,x3,x4 in zip(tmp1[::2], tmp1[1::2], range(1,num_cumulant+1,2), tableau_colorblind):
    ax.plot(disorder_strength, x1, color=x4, label='T^{}'.format(x3))
    ax.plot(disorder_strength, x2*linear_factor, 'x', color=x4, label='aT^{}'.format(x3+1))
ax.set_yscale('log')
ax.legend()
ax.set_xlabel('disorder strength Delta, m in [1-Delta,1+Delta]')
ax.set_ylabel('cumulant with h*omega=1')
