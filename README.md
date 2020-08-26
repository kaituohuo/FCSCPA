# Full Counting Statistics using Coherent Potential Approximation

keywords: coherent potential approximation (CPA), full couting statistics (FCS) theory, cumulant generating function (CGF), single site approximation (SSA), matrix inverse average, Broyden's method, self-consistent calculation, recursive Green's function (RGF) algorithm

## CPA

Given a complex matrix A of size $N\times N$, and a random diagonal matrix B of the same size with some specified distribution (e.g. uniform, binary) (all diagonal elements of B are drawed independently from the distribution). The CPA algorithm is able to calculate the matrix inverse average $\langle \frac{1}{A-B} \rangle$.

In quantum transport field, CPA algorithm is usually used to calculate the Green's function and related quantities (transmission, shot noise, etc.) (see reference below).

## quickstart

install

```bash
pip install git+https://github.com/husisy/FCSCPA.git
```

usage

```Python
from FCSCPA import CPA_general, CPA_block_general
from FCSCPA import FCS_CPA_oneband, FCS_CPA_multiband
from FCSCPA import generate_binary_hf_rand, generate_binary_hf_average
from FCSCPA import generate_uniform_hf_rand, generate_uniform_hf_average
from FCSCPA import generate_hf_Bii, Aii_to_matA
```

all functions provided in `FCSCPA` module

1. `CPA_general()`: CPA algorithm
2. `CPA_block_general()`: CPA algorithm when the random matrix B is **Block** diagonal
3. `broyden_solver()`: Broyden' method to solve the self-consistent equation `f(x)=x`
4. `np_inv_along_axis12()`: matrix inverse for high-dimensional (`np.ndim>2`) tensor
5. `matA_to_Aii()` and `Aii_to_matA()`: utils for block diagonal matrix
6. `generate_binary_hf_rand()` and `generate_uniform_hf_rand()`: (meta function) to generate value with some specified distribution
7. `generate_binary_hf_average()` and `generate_uniform_hf_average()`: (meta function) to evaluate the function average with some specified distribution
8. `generate_hf_Bii()`: (meta function), mapping single random variable to a random matrix
9. `FCS_CPA_oneband()`
10. `FCS_CPA_multiband()`
11. `ball_in_bowl()`
12. `Ymn_Leibniz_matAB()`
13. `Ymn_Leibniz()`
14. `demo_Ymn_Leibniz()`
15. `demo_Ymn_Binomial()`
16. `Ymn_electron_zeroK()`

see reference below for the algorithm detail. More examples will be added sooner, currently please see unittest scripts for detailed usage.

for developer

1. clone the entire repository `git clone https://github.com/husisy/FCSCPA.git`
2. install with editable mode `pip install -e .`
3. run the unittest `pytest -v .`
   * take about one minutes
   * sometimes some unittests may fail, since random number is used, it's okay to just run again (smile~)

## reference

Broyden's method to solve self-consistent equation

1. [A class of methods for solving nonlinear simultaneous equations](https://doi.org/10.1090/S0025-5718-1965-0198670-6)
2. *TODO add more ref*

coherent potential approximation

1. [Full counting statistics of conductance for disordered systems](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.115410)
2. *TODO add more ref*

full counting statistics theory

1. [Full counting statistics of conductance for disordered systems](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.96.115410)
2. [Full counting statistics of charge transport for disordered systerms](http://hub.hku.hk/handle/10722/265313)

## TODO

1. [x] move `FCS_CPA` folder here
2. [x] replace `.CPA` with `CPA`
3. [x] merge `CPA` with `FCSCPA`
4. [ ] apply FCSCPA to system in finite temperature
5. [ ] apply FCSCPA to two-sites system
6. [ ] TODO, converge issue, can we start from hermitian point
