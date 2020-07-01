import numpy as np
import scipy.constants
from tqdm import tqdm
import matplotlib.pyplot as plt
plt.ion()

from RGF import inv_tridiagonal_matrix, build_device_Hamiltonian_with_ham0_ham1
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


def hf_distribution_function(temperature_kelvin, energy_eV, chemical_potential_eV):
    tmp1 = ((energy_eV - chemical_potential_eV)
                * scipy.constants.elementary_charge
                / (scipy.constants.Boltzmann * temperature_kelvin))
    ret = 1 / (np.exp(tmp1) + 1)
    return ret


def linear_factor_small_potential_difference_limit(chemical_potential_left_eV,
                chemical_potential_right_eV, temperature_kelvin):
    assert chemical_potential_left_eV > chemical_potential_right_eV
    tmp1 = (chemical_potential_left_eV - chemical_potential_right_eV) * scipy.constants.elementary_charge
    tmp2 = 2 * scipy.constants.Boltzmann * temperature_kelvin
    ret = tmp1 / tmp2
    return ret


## parameter
num_layer = 2
NAPL = 4 #Number of Atoms Per Layer
num_cumulant = 8 #the cumulant_coeff will diverge for num_cumulant>=10 (2147483647)
EFNN = -2.6 #Energy of First Nearest Neighbor
EOnsite_central = 0
EOnsite_left = 0
EOnsite_right = 0
chemical_potential_left = 0.5003
chemical_potential_right = 0.4997
temperature = 100 #kelvin
EDisorder = np.linspace(0.01, 1)
# hf_average = generate_binary_hf_average(-EDisorder, 0.5, EDisorder)
hf_Bii = generate_hf_Bii(np.array([[0,1],[1,0]]))
energy = 0.5
energy_epsj = 1e-7j

fL = hf_distribution_function(temperature, energy, chemical_potential_left)
fR = hf_distribution_function(temperature, energy, chemical_potential_right)
linear_factor = linear_factor_small_potential_difference_limit(chemical_potential_left, chemical_potential_right, temperature)

# Ymn_matAB = [[Ymn_Leibniz_matAB(m,n) for m in range(1,n+1)] for n in range(1, num_cumulant+1)]
cumulant_coeff = []
for n in range(1, num_cumulant+1):
    tmp1 = []
    for m in range(1, n+1):
        matA,matB = Ymn_Leibniz_matAB(m, n)
        tmp1.append(Ymn_Leibniz(fL, fR, matA, matB, 'electron') * (1/(2*m)))
    cumulant_coeff.append(np.array(tmp1))

## calculation
NBPL = NAPL * 2
ham0 = EFNN*(np.diag(np.ones(NAPL-1),1) + np.diag(np.ones(NAPL-1),-1)) + EOnsite_left*np.eye(NAPL)
ham0 = np.kron(ham0, np.eye(2))
ham1 = EFNN * np.eye(NAPL)
ham1 = np.kron(ham1, np.eye(2))
HInfo = build_device_Hamiltonian_with_ham0_ham1(ham0, ham1, num_layer)

energyEye1 = (energy+energy_epsj)*np.eye(HInfo['ham_left_intra'].shape[0])

tmp1 = inv_tridiagonal_matrix(-HInfo['ham_left_inter'], energyEye1-HInfo['ham_left_intra'], -hfH(HInfo['ham_left_inter']))
self_energy_left = HInfo['ham_central_left_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_left_part']))
line_width_left = 1j * (self_energy_left - hfH(self_energy_left))
line_width_left = np.pad(line_width_left, [(0,NBPL*(num_layer-1)), (0,NBPL*(num_layer-1))], mode='constant')

tmp1 = inv_tridiagonal_matrix(-HInfo['ham_right_inter'], energyEye1-HInfo['ham_right_intra'], -hfH(HInfo['ham_right_inter']))
self_energy_right = HInfo['ham_central_right_part'] @ np.linalg.solve(tmp1, hfH(HInfo['ham_central_right_part']))
line_width_right = 1j * (self_energy_right - hfH(self_energy_right))
line_width_right = np.pad(line_width_right, [(NBPL*(num_layer-1),0), (NBPL*(num_layer-1),0)], mode='constant')

ret = []
inv_Green_part = (energy+energy_epsj)*np.eye(HInfo['ham_central'].shape[0]) - HInfo['ham_central']
inv_Green_part[:NBPL,:NBPL] -= self_energy_left
inv_Green_part[(-NBPL):,(-NBPL):] -= self_energy_right

cumulant = []
for EDisorder_i in tqdm(EDisorder):
    hf_average = generate_binary_hf_average(-EDisorder_i, 0.5, EDisorder_i)
    N_Gamma_trace = FCS_CPA_multiband(num_cumulant, inv_Green_part, -line_width_right, line_width_left, hf_Bii, hf_average)
    tmp1 = np.array([(1-2*(ind1%2))*x/2 for ind1,x in enumerate(N_Gamma_trace)])
    tmp1 = np.array([np.dot(x, N_Gamma_trace[:x.shape[0]]) for x in cumulant_coeff])
    if np.abs(tmp1.imag).max()>1e-5:
        print('[WARNING] large image part "{}j" for EDisorder_i={}'.format(np.abs(tmp1.imag).max(), EDisorder_i))
    cumulant.append(tmp1)


## figure
fig,ax = plt.subplots(1,1)
tmp1 = np.array(cumulant).T.real
tableau_colorblind = [x['color'] for x in plt.style.library['tableau-colorblind10']['axes.prop_cycle']]
for x1,x2,x3,x4 in zip(tmp1[::2], tmp1[1::2], range(1,num_cumulant+1,2), tableau_colorblind):
    ax.plot(EDisorder, x1, color=x4, label='$tr(T^{})$'.format(x3))
    ax.plot(EDisorder, x2*linear_factor, 'x', color=x4, label='$tr(aT^{})$'.format(x3+1))
ax.legend()
ax.set_xlabel('disorder strength Delta, onsite potential in [-w,+w]')
ax.set_ylabel('cumulant')
