import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

warnings.filterwarnings("ignore")

# setup variables 
lambda_bs = 0.0001 
sim_radius = 5000 
alpha = 4.0 
pt = 1.0 

# ! unused block, pls skip: 
# -------------------------------------------------
# # pre-calculate arrays for speed
# r_array = np.linspace(0.01, sim_radius, 5000)
# dr = r_array[1] - r_array[0]
# l_r_array = np.minimum(1.0, r_array**-alpha)
# r_dr_array = r_array * dr
# -------------------------------------------------

def char_func(t):
    
    # first integral: analyticla solution for r = 0 to 1 (where path loss l(r) = 1)
    part1_val = 0.5 * (1.0 - (1.0 / (1.0 - 1j * t * pt)))
    
    # second integral: numerical integral for r = 1 to infinity (where path loss l(r) = r^-alpha)
    def inner_integrand(r):
        return (1.0 - (1.0 / (1.0 - 1j * t * pt * (r**-alpha)))) * r
        
    # integrating to np.inf
    part2_val, _ = integrate.quad(inner_integrand, 1.0, np.inf, complex_func=True, epsabs=1e-5, epsrel=1e-5, limit=100)
    
    total_val = part1_val + part2_val
    return np.exp(-2.0 * np.pi * lambda_bs * total_val)

def gil_pelaez_infinite(T_watts):
    # expanding Im[phi * e^(-j*t*T)] into real/imag components
    # to utilize QAWF algorithm for infinite oscillatory tails (bounds of which are [1e-7, np.inf])
    def integrand_imag(t):
        return np.imag(char_func(t)) / t
        
    def integrand_real(t):
        return np.real(char_func(t)) / t
        
    tolerance = 1e-4 
    
    # changed 'w' to 'wvar' for the scipy syntax to work
    res_cos, _ = integrate.quad(integrand_imag, 1e-7, np.inf, weight='cos', wvar=T_watts, epsabs=tolerance, epsrel=tolerance, limit=150)
    res_sin, _ = integrate.quad(integrand_real, 1e-7, np.inf, weight='sin', wvar=T_watts, epsabs=tolerance, epsrel=tolerance, limit=150)
    
    return 0.5 - (1.0 / np.pi) * (res_cos - res_sin)

# x axis thresholds in dBm (let's say -50 dBm to 0 dBm)
ipd_thresholds_dbm = np.linspace(-50, 0, 16)
theoretical_cdf = []

print("calculating mathematical derivation... please wait...")

for t_dbm in ipd_thresholds_dbm:
    # reverse formula: convert dBm back to Watts for the math function
    t_watts = 10 ** ((t_dbm - 30) / 10.0)
    
    prob = gil_pelaez_infinite(t_watts)
    theoretical_cdf.append(prob)

plt.figure(figsize=(8, 6))
# red dashed line with triangles on every point
plt.plot(ipd_thresholds_dbm, theoretical_cdf, label='analytical math (Eqn 2.32)', color='red', marker='^', linestyle='--') # , markevery =2) 

plt.xlabel('ipd test values (dBm)')
plt.ylabel('cdf')
plt.title('analytical Graph for Total Exposure')
plt.legend()
plt.grid(True)
plt.xlim(right = 0)
plt.show()