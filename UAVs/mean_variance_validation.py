import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

warnings.filterwarnings("ignore")

print("Setting up 3D UAV network parameters...")
num_sims = 10000 
lambda_R = 5e-5     # UAV density
H_R = 100.0         # Constant altitude of the UAV fleet in meters
sim_radius = 15000  # Large radius to approximate infinite plane
P_R = 1.0           # Transmit power in watts

# LoS Propagation Parameters (Nakagami m > 1)
alpha_L = 2.5       
beta_L = 100.0      
m_L = 3.0           

# NLoS Propagation Parameters (Nakagami m = 1 -> Rayleigh)
alpha_N = 4.0       
beta_N = 1000.0     
m_N = 1.0           

# AL HOURANI'S LoS PROBABILITY MODEL
# Standard Urban environment parameters from Al Hourani's paper
env_a = 9.61
env_b = 0.16

def p_AL(r):
    # Calculate elevation angle in degrees. 
    # If r is 0 (drone directly overhead), angle is 90 degrees.
    theta = np.where(r == 0, 90.0, np.degrees(np.arctan(H_R / r)))
    return 1.0 / (1.0 + env_a * np.exp(-env_b * (theta - env_a)))

def p_AN(r):
    return 1.0 - p_AL(r)


# MONTE CARLO SIMULATION
print(f"Simulating {num_sims} UAV networks using Al Hourani's model...")
simulated_E_totals = []

for _ in range(num_sims):
    # Poisson Point Process generation
    area = np.pi * (sim_radius ** 2)
    num_uavs = np.random.poisson(lambda_R * area)
    if num_uavs == 0: continue 
    
    r_coords = sim_radius * np.sqrt(np.random.uniform(0, 1, num_uavs))
    d_3D = np.sqrt(r_coords**2 + H_R**2)
    
    # Thinning Theorem utilizing Al Hourani probabilities
    rand_probs = np.random.uniform(0, 1, num_uavs)
    is_LoS = rand_probs < p_AL(r_coords)
    is_NLoS = ~is_LoS
    
    # Calculate Received Powers (Gamma power distributions)
    E_L = 0
    if np.any(is_LoS):
        d_LoS = d_3D[is_LoS]
        h_L = np.random.gamma(shape=m_L, scale=1.0/m_L, size=len(d_LoS))
        S_L = (P_R * h_L) / (beta_L * d_LoS**alpha_L)
        E_L = np.sum(S_L)
        
    E_N = 0
    if np.any(is_NLoS):
        d_NLoS = d_3D[is_NLoS]
        h_N = np.random.gamma(shape=m_N, scale=1.0/m_N, size=len(d_NLoS))
        S_N = (P_R * h_N) / (beta_N * d_NLoS**alpha_N)
        E_N = np.sum(S_N)
        
    simulated_E_totals.append(E_L + E_N)

sim_mean_E = np.mean(simulated_E_totals)
sim_var_E = np.var(simulated_E_totals)


# ANALYTICAL MATH, CAMPBELL's THEOREM
print("Computing analytical closed-form integrals...")

def mean_integrand(r):
    los_comp = p_AL(r) / (beta_L * (r**2 + H_R**2)**(alpha_L / 2.0))
    nlos_comp = p_AN(r) / (beta_N * (r**2 + H_R**2)**(alpha_N / 2.0))
    return (los_comp + nlos_comp) * r

def var_integrand(r):
    los_comp = (p_AL(r) * (m_L + 1)) / (m_L * (beta_L**2) * (r**2 + H_R**2)**alpha_L)
    nlos_comp = (p_AN(r) * (m_N + 1)) / (m_N * (beta_N**2) * (r**2 + H_R**2)**alpha_N)
    return (los_comp + nlos_comp) * r

ana_mean_E_val, _ = integrate.quad(mean_integrand, 0, np.inf, epsabs=1e-6, epsrel=1e-6)
ana_mean_E = 2 * np.pi * lambda_R * P_R * ana_mean_E_val

ana_var_E_val, _ = integrate.quad(var_integrand, 0, np.inf, epsabs=1e-6, epsrel=1e-6)
ana_var_E = 2 * np.pi * lambda_R * (P_R**2) * ana_var_E_val

mean_error = abs(sim_mean_E - ana_mean_E) / ana_mean_E * 100
var_error = abs(sim_var_E - ana_var_E) / ana_var_E * 100


# VISUALIZATION DASHBOARD
print("Generating visualization dashboard...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
fig.patch.set_facecolor('#f4f4f4')

metrics = ['Simulation', 'Analytical Math']
colors = ['#1f77b4', '#ff7f0e']

# Plot 1
bars1 = ax1.bar(metrics, [sim_mean_E, ana_mean_E], color=colors, edgecolor='black', width=0.6)
ax1.set_title('Expected Mean Exposure $\mathbb{E}[E]$', fontsize=14, fontweight='bold')
ax1.set_ylabel('Absorbed Power (Watts)', fontsize=12)
ax1.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3e}', ha='center', va='bottom', fontsize=11)
ax1.text(0.5, 0.9, f'Error: {mean_error:.4f}%', transform=ax1.transAxes, ha='center', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=12, fontweight='bold')

# Plot 2
bars2 = ax2.bar(metrics, [sim_var_E, ana_var_E], color=colors, edgecolor='black', width=0.6)
ax2.set_title('Exposure Variance Var$[E]$', fontsize=14, fontweight='bold')
ax2.set_ylabel('Power Variance (Watts$^2$)', fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.7)

for bar in bars2:
    yval = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.3e}', ha='center', va='bottom', fontsize=11)
ax2.text(0.5, 0.9, f'Error: {var_error:.4f}%', transform=ax2.transAxes, ha='center', 
         bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'), fontsize=12, fontweight='bold')

plt.suptitle("Validation of UAV Network Exposure (Al Hourani Model)", fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

print("Validation complete!")