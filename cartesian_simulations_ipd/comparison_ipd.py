import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

# ignore scipy warnings if the fourier tails get too microscopically small
warnings.filterwarnings("ignore")

# setup variables
number_of_simulations = 10000 # run 10k times
lambda_bs = 0.0001 # density
sim_radius = 5000 # keeping MC simulation at 5km so it actually finishes running
alpha = 4.0 # path loss exponent
pt = 1.0 # transmit power in watts

simulated_ipd_watts = []
print("simulating 10,000 networks... this might take a minute...")

# MONTE CARLO SIMULATION
for _ in range(number_of_simulations):
    # generate area and number of stations
    area = (2 * sim_radius) ** 2
    num_bs = np.random.poisson(lambda_bs * area)
    if num_bs == 0: continue 
    
    # cartesian coords uniformly distributed
    x_coords = np.random.uniform(-sim_radius, sim_radius, num_bs)
    y_coords = np.random.uniform(-sim_radius, sim_radius, num_bs)
    
    # pythagorean distance
    distances = np.sqrt(x_coords**2 + y_coords**2)
    
    # crop to the circle limit to prevent clumping at the edges
    valid = (distances <= sim_radius)
    distances = distances[valid]
    if len(distances) == 0: continue
    
    # BOUNDED path loss: min(1, r^-alpha)
    # stops the divide-by-zero singularity if a tower is at r=0
    path_loss = np.minimum(1.0, distances**-alpha)
    
    # rayleigh fading for all channels
    fading = np.random.exponential(1.0, size=len(distances))
    
    # IPD is the sum of EVERYTHING
    received_powers = pt * fading * path_loss 
    total_ipd = np.sum(received_powers)
    
    simulated_ipd_watts.append(total_ipd)

# turn 10k results into a CDF curve
simulated_ipd_watts = np.sort(simulated_ipd_watts)
sim_cdf = np.arange(1, len(simulated_ipd_watts) + 1) / len(simulated_ipd_watts)

# convert simulation watts to dBm for the graph
simulated_ipd_dbm = 10 * np.log10(simulated_ipd_watts) + 30

print("calculating infinite nested integrals... heavily relying on scipy quadrature...")

# MATH FORMULA (Eq 2.19 & 2.32)

def char_func(t):
    # Integrating strictly to np.inf.
    # Path loss is piecewise: l(r) = 1 for r<=1, and r^-alpha for r>1.
    # So we split the integral at r=1 to help the numerical quadrature.
    
    # Part 1: r from 0 to 1
    # Analytical solution for int_0^1 (1 - 1/(1 - j*t*pt)) * r dr
    part1_val = 0.5 * (1.0 - (1.0 / (1.0 - 1j * t * pt)))
    
    # Part 2: r from 1 to infinity
    def inner_integrand(r):
        return (1.0 - (1.0 / (1.0 - 1j * t * pt * (r**-alpha)))) * r
        
    # Using epsabs and eprel for accuracy
    part2_val, _ = integrate.quad(inner_integrand, 1.0, np.inf, complex_func=True, epsabs=1e-5, epsrel=1e-5, limit=100)
    
    total_val = part1_val + part2_val
    return np.exp(-2.0 * np.pi * lambda_bs * total_val)

def gil_pelaez_infinite(T_watts):
    # Standard quad chokes on highly oscillatory infinite integrals.
    # So algebraically expanded Im[phi * e^(-j*t*T)] into sines and cosines
    # so SciPy's QAWF Fourier algorithm can be triggered using weight parameters.
    
    def integrand_imag(t):
        return np.imag(char_func(t)) / t
        
    def integrand_real(t):
        return np.real(char_func(t)) / t
        
    tolerance = 1e-4 # Error tolerances for the Fourier tails
    
    # Started at 1e-7 to avoid divide by zero pole at t=0
    # weight='cos' and 'sin' ask the Fortran backend to handle the infinite oscillations
    # BUG Fix: Changed 'w' to 'wvar' for the scipy syntax
    res_cos, _ = integrate.quad(integrand_imag, 1e-7, np.inf, weight='cos', wvar=T_watts, epsabs=tolerance, epsrel=tolerance, limit=150)
    res_sin, _ = integrate.quad(integrand_real, 1e-7, np.inf, weight='sin', wvar=T_watts, epsabs=tolerance, epsrel=tolerance, limit=150)
    
    total_integral = res_cos - res_sin
    return 0.5 - (1.0 / np.pi) * total_integral

# pick 15 test thresholds (fewer points because nested infinite integrals are computationally heavy)
min_w = np.min(simulated_ipd_watts)
max_w = 0.001 # np.max(simulated_ipd_watts)
target_thresholds_watts = np.logspace(np.log10(min_w), np.log10(max_w), 15)

analytical_cdf = []
analytical_dbm = []

for T in target_thresholds_watts:
    prob = gil_pelaez_infinite(T)
    analytical_cdf.append(prob)
    analytical_dbm.append(10 * np.log10(T) + 30)

# plot the two on top of each other
plt.figure(figsize=(9, 6))

plt.plot(simulated_ipd_dbm, sim_cdf, label='Simulation (10k random networks)', color='blue', linewidth=2)
plt.plot(analytical_dbm, analytical_cdf, label='The Math (Eq 2.32)', color='red', marker='^', linestyle='--')

plt.xlabel('Incident Power Density (dBm)')
plt.ylabel('Cumulative Distribution Function (CDF)')
plt.title('Validation of IPD EMF Exposure (Simulation vs Analytical)')
plt.legend()
plt.grid(True)
plt.xlim(right=0)
plt.show()