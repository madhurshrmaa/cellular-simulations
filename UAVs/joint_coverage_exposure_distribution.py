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

print("starting the joint metric monte carlo simulation now... running 10000 networks...")

sim_sinr_db = []
sim_exp_dbm = []

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
        is_serving_los = True
        h_0 = np.random.gamma(m_l, 1.0/m_l)
        serving_power = (p_r * h_0) / (beta_l * d_0**alpha_l)
    else:
        is_serving_los = False
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
        
    # exposure is the total power absorbed so serving plus interference
    total_exposure = serving_power + interference
    sinr = serving_power / (interference + noise_w)
    
    if sinr > 0 and total_exposure > 0:
        sim_sinr_db.append(10 * np.log10(sinr))
        sim_exp_dbm.append(10 * np.log10(total_exposure) + 30)


print("doing the heavy math now... calculating frechet bounds so we dont have to do the massive joint integral...")

# coverage math
# real and imag splits
def pdf_r0(r0):
    return 2 * np.pi * lambda_r * r0 * np.exp(-np.pi * lambda_r * r0**2)

def laplace_serving(s, r0, is_los):
    if is_los:
        return (1.0 + (s * p_r) / (m_l * beta_l * (r0**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
    else:
        return (1.0 + (s * p_r) / (m_n * beta_n * (r0**2 + h_r**2)**(alpha_n/2.0))) ** -m_n

def laplace_interference_los(s, r0):
    def integrand_real(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        return np.real(p_al(r) * term * r)
    def integrand_imag(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        return np.imag(p_al(r) * term * r)
    
    val_r, _ = integrate.quad(integrand_real, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    val_i, _ = integrate.quad(integrand_imag, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

def laplace_interference_nlos(s, r0):
    def integrand_real(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        return np.real(p_an(r) * term * r)
    def integrand_imag(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        return np.imag(p_an(r) * term * r)
        
    val_r, _ = integrate.quad(integrand_real, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    val_i, _ = integrate.quad(integrand_imag, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

scale_p = p_r / (beta_l * (h_r**alpha_l)) 

def gil_pelaez_coverage(t_linear, r0, is_los):
    def gp_integrand(v):
        u = v / scale_p
        lap_s = laplace_serving(-1j * u, r0, is_los)
        lap_il = laplace_interference_los(1j * u * t_linear, r0)
        lap_in = laplace_interference_nlos(1j * u * t_linear, r0)
        phi_z = lap_s * lap_il * lap_in
        return np.imag(phi_z * np.exp(-1j * u * t_linear * noise_w)) / v
        
    val, _ = integrate.quad(gp_integrand, 1e-6, np.inf, epsabs=1e-3, epsrel=1e-3, limit=100)
    return 0.5 + (1.0 / np.pi) * val


# exposure math
# bug fix: negative sign fix and the 1e-12 epsabs
def laplace_exposure(s):
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

    val_r, _ = integrate.quad(integrand_real, 0, sim_radius, epsabs=1e-12, epsrel=1e-4, limit=100)
    val_i, _ = integrate.quad(integrand_imag, 0, sim_radius, epsabs=1e-12, epsrel=1e-4, limit=100)
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

def gil_pelaez_exposure(theta_watts):
    def gp_integrand(v):
        u = v / theta_watts
        # keeping the typo fix here forever so it doesnt break again
        lap_e = laplace_exposure(-1j * u) 
        return np.imag(lap_e * np.exp(-1j * v)) / v
        
    val, _ = integrate.quad(gp_integrand, 1e-9, 2000, epsabs=1e-5, epsrel=1e-4, limit=200)
    return 0.5 - (1.0 / np.pi) * val


# building the 15x15 grids
print("setting up the frechet grids... this calculates individual arrays then combines them...")

t_db_vals = np.linspace(-20, 20, 15)

min_dbm = np.min(sim_exp_dbm)
max_dbm = np.max(sim_exp_dbm)
exp_dbm_vals = np.linspace(min_dbm, max_dbm, 15)

# crunching the coverage array just once
p_cov_arr = []
for t_db in t_db_vals:
    t_linear = 10**(t_db / 10.0)
    def outer_integral(r0):
        prob_los = p_al(r0) * gil_pelaez_coverage(t_linear, r0, True)
        prob_nlos = p_an(r0) * gil_pelaez_coverage(t_linear, r0, False)
        return pdf_r0(r0) * (prob_los + prob_nlos)
        
    final_prob, _ = integrate.quad(outer_integral, 0, 1000, epsabs=1e-3, epsrel=1e-3, limit=50)
    p_cov_arr.append(final_prob)
    
# crunching the exposure array just once
p_exp_arr = []
for t_dbm in exp_dbm_vals:
    t_watts = 10**((t_dbm - 30) / 10.0)
    prob = gil_pelaez_exposure(t_watts)
    p_exp_arr.append(prob)

# now doing the actual matrices
X, Y = np.meshgrid(t_db_vals, exp_dbm_vals)
sim_matrix = np.zeros_like(X)
ana_matrix = np.zeros_like(X)

for i in range(15):
    for j in range(15):
        t_db_target = t_db_vals[j]
        exp_dbm_target = exp_dbm_vals[i]
        
        # monte carlo simulation joint probability
        # literally just counting if both things are true at the same time
        success_count = 0
        for k in range(len(sim_sinr_db)):
            if sim_sinr_db[k] > t_db_target and sim_exp_dbm[k] < exp_dbm_target:
                success_count = success_count + 1
                
        sim_matrix[i, j] = success_count / len(sim_sinr_db)
        
        # upper frechet bound approximation dr wiame asked for
        # perfectly bounds the joint probability just by taking the min of the two separate arrays
        ana_matrix[i, j] = min(p_cov_arr[j], p_exp_arr[i])
        
diff_matrix = np.abs(sim_matrix - ana_matrix)

print("plotting the heatmaps side by side...")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# fig 1: sim
c1 = axes[0].pcolormesh(X, Y, sim_matrix, shading='gouraud', cmap='viridis', vmin=0, vmax=1)
axes[0].set_title("Simulation (10,000 MC Networks)")
axes[0].set_xlabel("Coverage threshold T [dB]")
axes[0].set_ylabel("Exposure threshold T' [dBm]")
fig.colorbar(c1, ax=axes[0])

# fig 2: analytical frechet
c2 = axes[1].pcolormesh(X, Y, ana_matrix, shading='gouraud', cmap='viridis', vmin=0, vmax=1)
axes[1].set_title("Upper Fréchet Bound Approx")
axes[1].set_xlabel("Coverage threshold T [dB]")
fig.colorbar(c2, ax=axes[1])

# fig 3: difference
c3 = axes[2].pcolormesh(X, Y, diff_matrix, shading='gouraud', cmap='viridis', vmin=0, vmax=0.05)
axes[2].set_title("Absolute Difference |Sim - Bound|")
axes[2].set_xlabel("Coverage threshold T [dB]")
fig.colorbar(c3, ax=axes[2])

plt.suptitle("Validation of Joint Coverage-Exposure Distribution J_lim(T, T')", fontsize=16)
plt.tight_layout()
plt.show()