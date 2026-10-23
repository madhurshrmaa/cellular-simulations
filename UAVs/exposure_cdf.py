import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

# turning off warnings so the terminal stays clean
warnings.filterwarnings("ignore")

num_sims = 10000 
lambda_r = 5e-5 
h_r = 100.0 
sim_radius = 5000 
p_r = 1.0 

# los parameters
alpha_l = 2.5       
beta_l = 100.0      
m_l = 3.0           

# nlos rayleigh parameters
alpha_n = 4.0       
beta_n = 1000.0     
m_n = 1.0           

# al hourani urban parameters
env_a = 9.61
env_b = 0.16

def p_al(r):
    # finding the elevation angle
    if r == 0:
        theta = 90.0
    else:
        theta = np.degrees(np.arctan(h_r / r))
        
    return 1.0 / (1.0 + env_a * np.exp(-env_b * (theta - env_a)))

def p_an(r):
    # just basic complement
    return 1.0 - p_al(r)

print("running monte carlo simulation for exposure... dropping 10000 networks...")
simulated_e_totals = []

for i in range(num_sims):
    area = np.pi * (sim_radius ** 2)
    num_uavs = np.random.poisson(lambda_r * area)
    
    if num_uavs == 0: 
        continue 
        
    r_coords = sim_radius * np.sqrt(np.random.uniform(0, 1, num_uavs))
    d_3d = np.sqrt(r_coords**2 + h_r**2)
    
    total_exposure = 0.0
    
    for j in range(num_uavs):
        r_drone = r_coords[j]
        d_drone = d_3d[j]
        
        if np.random.uniform(0, 1) < p_al(r_drone):
            is_los = True
        else:
            is_los = False
            
        if is_los:
            h_fading = np.random.gamma(m_l, 1.0/m_l)
            received_power = (p_r * h_fading) / (beta_l * d_drone**alpha_l)
        else:
            h_fading = np.random.gamma(m_n, 1.0/m_n)
            received_power = (p_r * h_fading) / (beta_n * d_drone**alpha_n)
            
        total_exposure = total_exposure + received_power
        
    simulated_e_totals.append(total_exposure)

# sorting it to build the cdf line
simulated_e_totals = np.sort(simulated_e_totals)
sim_e_dbm = 10 * np.log10(simulated_e_totals) + 30
sim_cdf = np.arange(1, len(simulated_e_totals) + 1) / len(simulated_e_totals)


print("doing the math... going back to foundational integration methods with the negative sign fixed...")

def laplace_exposure(s):
    # manually splitting real and imaginary parts using standard integration
    def integrand_real(r):
        term_l = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        val_l = p_al(r) * term_l * r
        
        term_n = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        val_n = p_an(r) * term_n * r
        
        return np.real(val_l + val_n)
        
    def integrand_imag(r):
        term_l = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        val_l = p_al(r) * term_l * r
        
        term_n = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        val_n = p_an(r) * term_n * r
        
        return np.imag(val_l + val_n)

    # using epsabs=1e-12 so scipy doesnt skip the tiny powers and flatline the integral
    val_r, _ = integrate.quad(integrand_real, 0, sim_radius, epsabs=1e-12, epsrel=1e-4, limit=100)
    val_i, _ = integrate.quad(integrand_imag, 0, sim_radius, epsabs=1e-12, epsrel=1e-4, limit=100)
    
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

def gil_pelaez_exposure(theta_watts):
    def gp_integrand(v):
        # standard substitution to normalize frequency
        u = v / theta_watts
        
        # the crucial negative sign fix that broke everything earlier
        lap_e = laplace_exposure(-1j * u)
        return np.imag(lap_e * np.exp(-1j * v)) / v
        
    val, _ = integrate.quad(gp_integrand, 1e-9, 2000, epsabs=1e-5, epsrel=1e-4, limit=200)
    return 0.5 - (1.0 / np.pi) * val

# getting 15 threshold points
min_w = np.min(simulated_e_totals)
max_w = np.max(simulated_e_totals)

math_theta_watts = np.logspace(np.log10(min_w), np.log10(max_w), 15)

math_cdf = []
math_dbm = []

for t_watts in math_theta_watts:
    print(f"crunching math for {10 * np.log10(t_watts) + 30:.1f} dbm...")
    prob = gil_pelaez_exposure(t_watts)
    math_cdf.append(prob)
    math_dbm.append(10 * np.log10(t_watts) + 30)

print("plotting the final exposure cdf...")

plt.figure(figsize=(9, 6))

plt.plot(sim_e_dbm, sim_cdf, label='Simulation (10,000 drops)', color='blue', linewidth=2)
plt.plot(math_dbm, math_cdf, label='Analytical Math (Gil-Pelaez)', color='red', marker='^', linestyle='--')

plt.xlabel('Total Absorbed Exposure (dBm)')
plt.ylabel('Cumulative Distribution Function F_E(\u03B8)')
plt.title('Validation of UAV Network Exposure CDF')
plt.legend()
plt.grid(True)
plt.xlim(np.min(sim_e_dbm), np.max(sim_e_dbm))
plt.ylim(0, 1.05)
plt.show()