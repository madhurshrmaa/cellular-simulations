import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

# shutting up the warnings so the console is clean
warnings.filterwarnings("ignore")

num_sims = 10000 
lambda_r = 5e-5 
h_r = 100.0 
sim_radius = 5000 
p_r = 1.0 
noise_w = 1e-12 

# los vars
alpha_l = 2.5       
beta_l = 100.0      
m_l = 3.0           

# nlos rayleigh vars
alpha_n = 4.0       
beta_n = 1000.0     
m_n = 1.0           

# urban environment
env_a = 9.61
env_b = 0.16

def p_al(r):
    # find the angle looking up at the drone
    if r == 0:
        theta = 90.0
    else:
        theta = np.degrees(np.arctan(h_r / r))
        
    return 1.0 / (1.0 + env_a * np.exp(-env_b * (theta - env_a)))

def p_an(r):
    return 1.0 - p_al(r)

print("starting the data rate monte carlo simulation now... running 10000 networks...")

sim_capacity_arr = []

for i in range(num_sims):
    area = np.pi * (sim_radius ** 2)
    num_uavs = np.random.poisson(lambda_r * area)
    
    if num_uavs == 0: 
        continue 
        
    r_coords = sim_radius * np.sqrt(np.random.uniform(0, 1, num_uavs))
    d_3d = np.sqrt(r_coords**2 + h_r**2)
    
    closest_idx = np.argmin(d_3d)
    r_0 = r_coords[closest_idx]
    d_0 = d_3d[closest_idx]
    
    # serving link calculation
    if np.random.uniform(0, 1) < p_al(r_0):
        h_0 = np.random.gamma(m_l, 1.0/m_l)
        serving_power = (p_r * h_0) / (beta_l * d_0**alpha_l)
    else:
        h_0 = np.random.gamma(m_n, 1.0/m_n)
        serving_power = (p_r * h_0) / (beta_n * d_0**alpha_n)
        
    interference = 0.0
    
    # adding up all the interference from the other drones
    for j in range(num_uavs):
        if j == closest_idx:
            continue
            
        r_drone = r_coords[j]
        d_drone = d_3d[j]
        
        if np.random.uniform(0, 1) < p_al(r_drone):
            h_i = np.random.gamma(m_l, 1.0/m_l)
            interf_power = (p_r * h_i) / (beta_l * d_drone**alpha_l)
        else:
            h_i = np.random.gamma(m_n, 1.0/m_n)
            interf_power = (p_r * h_i) / (beta_n * d_drone**alpha_n)
            
        interference = interference + interf_power
        
    sinr = serving_power / (interference + noise_w)
    
    if sinr > 0:
        # shannon capacity in bits/s/Hz
        cap = np.log2(1.0 + sinr)
        sim_capacity_arr.append(cap)

sim_data_rate = np.mean(sim_capacity_arr)


print("doing the heavy math now... evaluating the integrals for capacity...")

# coverage math
def pdf_r0(r0):
    return 2 * np.pi * lambda_r * r0 * np.exp(-np.pi * lambda_r * r0**2)

# doing real valued laplace transforms here since the z variable is entirely real, unlike gil-pelaez
def laplace_serving_real(z, r0, is_los):
    if is_los:
        return (1.0 + (z * p_r) / (m_l * beta_l * (r0**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
    else:
        return (1.0 + (z * p_r) / (m_n * beta_n * (r0**2 + h_r**2)**(alpha_n/2.0))) ** -m_n

def laplace_interference_total_real(z, r0):
    def integrand_los(r):
        term = 1.0 - (1.0 + (z * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        return p_al(r) * term * r
        
    def integrand_nlos(r):
        term = 1.0 - (1.0 + (z * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        return p_an(r) * term * r
    
    val_l, _ = integrate.quad(integrand_los, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    val_n, _ = integrate.quad(integrand_nlos, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    
    return np.exp(-2 * np.pi * lambda_r * (val_l + val_n))

def expected_rate_cond(r0, is_los):
    def z_integrand(z):
        lap_i = laplace_interference_total_real(z, r0)
        lap_s = laplace_serving_real(z, r0, is_los)
        return (lap_i / z) * (1.0 - lap_s) * np.exp(-z * noise_w)
        
    # integrating starting from 1e-9 to strictly dodge the zero division error at z=0
    val, _ = integrate.quad(z_integrand, 1e-9, 2000, epsabs=1e-3, epsrel=1e-3, limit=100)
    
    # reverting the natural log base back to base 2 for bits/s/hz
    return val / np.log(2.0)

def outer_integral(r0):
    rate_los = p_al(r0) * expected_rate_cond(r0, True)
    rate_nlos = p_an(r0) * expected_rate_cond(r0, False)
    return pdf_r0(r0) * (rate_los + rate_nlos)

ana_data_rate, _ = integrate.quad(outer_integral, 0, 1500, epsabs=1e-3, epsrel=1e-3, limit=50)

print("plotting the validation bar chart now...")

fig, ax = plt.subplots(figsize=(8, 6))

labels = ['Simulation', 'Analytical Math']
values = [sim_data_rate, ana_data_rate]

bars = ax.bar(labels, values, color=['#1f77b4', '#ff7f0e'], edgecolor='black', width=0.6)

# error calc
error_pct = abs(sim_data_rate - ana_data_rate) / sim_data_rate * 100.0

# throwing the error box up top perfectly centered
ax.text(0.5, max(values) * 0.95, f"Error: {error_pct:.4f}%", 
        ha='center', va='top', fontsize=12, fontweight='bold',
        bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'))

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(values)*0.01), f"{yval:.4e}", 
            ha='center', va='bottom', fontsize=10)

ax.set_ylabel('Data Rate (bits/s/Hz)')
ax.set_title(r'Validation of Data Rate $\mathbb{E}[C]$')
ax.yaxis.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()