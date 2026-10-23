import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
import warnings

# ignoring warnings so the terminal doesnt get spammed when scipy hits a fourier tail
warnings.filterwarnings("ignore")

# setting up all the basic parameters for the drones and the environment
num_sims = 10000 
lambda_r = 5e-5 
h_r = 100.0 
sim_radius = 5000 
p_r = 1.0 
noise_w = 1e-12 # just some basic thermal noise

# los stuff
alpha_l = 2.5       
beta_l = 100.0      
m_l = 3.0           

# nlos stuff where m=1 makes it rayleigh fading
alpha_n = 4.0       
beta_n = 1000.0     
m_n = 1.0           

# al hourani urban environment parameters
env_a = 9.61
env_b = 0.16

def p_al(r):
    # figuring out the angle looking up at the drone
    # if r is 0 it means the drone is directly over our head so its 90 degrees
    if r == 0:
        theta = 90.0
    else:
        theta = np.degrees(np.arctan(h_r / r))
        
    # the math formula for al hourani
    return 1.0 / (1.0 + env_a * np.exp(-env_b * (theta - env_a)))

def p_an(r):
    # it has to be either los or nlos so just subtract from 1
    return 1.0 - p_al(r)


print("running monte carlo simulation now... going through 10000 random drops...")
simulated_sinr_list = []

for i in range(num_sims):
    # generating the area and finding out how many drones are inside it
    area = np.pi * (sim_radius ** 2)
    num_uavs = np.random.poisson(lambda_r * area)
    
    if num_uavs == 0: 
        continue 
        
    # putting the drones in random spots around the circle
    r_coords = sim_radius * np.sqrt(np.random.uniform(0, 1, num_uavs))
    
    # 3d distance because they fly at h_r altitude
    d_3d = np.sqrt(r_coords**2 + h_r**2)
    
    # finding the closest drone because that one is going to be the serving link
    closest_index = np.argmin(d_3d)
    r_0 = r_coords[closest_index]
    d_0 = d_3d[closest_index]
    
    # checking if the serving drone is los or nlos using a random dice roll
    if np.random.uniform(0, 1) < p_al(r_0):
        is_serving_los = True
    else:
        is_serving_los = False
    
    # getting the signal power from the serving drone using nakagami gamma distributions
    if is_serving_los:
        h_0 = np.random.gamma(m_l, 1.0/m_l)
        serving_power = (p_r * h_0) / (beta_l * d_0**alpha_l)
    else:
        h_0 = np.random.gamma(m_n, 1.0/m_n)
        serving_power = (p_r * h_0) / (beta_n * d_0**alpha_n)
        
    # now we need to calculate the interference from all the other drones
    interference = 0.0
    
    for j in range(num_uavs):
        # skip the serving drone so we dont count it as interference
        if j == closest_index:
            continue
            
        r_interf = r_coords[j]
        d_interf = d_3d[j]
        
        # roll dice for interference drone
        if np.random.uniform(0, 1) < p_al(r_interf):
            is_interf_los = True
        else:
            is_interf_los = False
        
        if is_interf_los:
            h_i = np.random.gamma(m_l, 1.0/m_l)
            interf_power = (p_r * h_i) / (beta_l * d_interf**alpha_l)
        else:
            h_i = np.random.gamma(m_n, 1.0/m_n)
            interf_power = (p_r * h_i) / (beta_n * d_interf**alpha_n)
            
        # add it to the total interference
        interference = interference + interf_power
        
    # sinr formula
    sinr = serving_power / (interference + noise_w)
    
    # convert to db so we can graph it easier
    if sinr > 0:
        sinr_db = 10 * np.log10(sinr)
        simulated_sinr_list.append(sinr_db)

# getting the simulation graph ready by testing a bunch of thresholds
thresholds_db_sim = np.linspace(-20, 20, 100)
sim_coverage_probs = []

for t_db in thresholds_db_sim:
    # counting how many times the network actually survived the threshold
    success_count = 0
    for sinr in simulated_sinr_list:
        if sinr > t_db:
            success_count = success_count + 1
            
    sim_coverage_probs.append(success_count / len(simulated_sinr_list))


print("calculating the heavy math chunk...")

def pdf_r0(r0):
    # probability density function of the closest drone
    return 2 * np.pi * lambda_r * r0 * np.exp(-np.pi * lambda_r * r0**2)

def laplace_serving(s, r0, is_los):
    # laplace transform for the serving signal from the paper
    if is_los:
        return (1.0 + (s * p_r) / (m_l * beta_l * (r0**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
    else:
        return (1.0 + (s * p_r) / (m_n * beta_n * (r0**2 + h_r**2)**(alpha_n/2.0))) ** -m_n

def laplace_interference_los(s, r0):
    # separating real and imaginary parts manually so scipy doesnt crash in the inner integral
    def integrand_real(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        return np.real(p_al(r) * term * r)
        
    def integrand_imag(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_l * beta_l * (r**2 + h_r**2)**(alpha_l/2.0))) ** -m_l
        return np.imag(p_al(r) * term * r)
    
    # integrate from r0 to sim radius because interference is always further away
    val_r, _ = integrate.quad(integrand_real, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    val_i, _ = integrate.quad(integrand_imag, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

def laplace_interference_nlos(s, r0):
    # doing the exact same manual split for the nlos side
    def integrand_real(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        return np.real(p_an(r) * term * r)
        
    def integrand_imag(r):
        term = 1.0 - (1.0 + (s * p_r) / (m_n * beta_n * (r**2 + h_r**2)**(alpha_n/2.0))) ** -m_n
        return np.imag(p_an(r) * term * r)
        
    val_r, _ = integrate.quad(integrand_real, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    val_i, _ = integrate.quad(integrand_imag, r0, sim_radius, epsabs=1e-3, epsrel=1e-3)
    
    return np.exp(-2 * np.pi * lambda_r * (val_r + 1j * val_i))

# bug fix: scaling variable so scipy doesnt miss the microscopic power changes
scale_p = p_r / (beta_l * (h_r**alpha_l)) 

def gil_pelaez_math(t_linear, r0, is_los):
    # mapping it all into the frequency domain
    def gp_integrand(v):
        # unscaling u here
        u = v / scale_p
        
        lap_s = laplace_serving(-1j * u, r0, is_los)
        lap_il = laplace_interference_los(1j * u * t_linear, r0)
        lap_in = laplace_interference_nlos(1j * u * t_linear, r0)
        
        phi_z = lap_s * lap_il * lap_in
        
        # the imaginary part of the formula dividing by v instead of u to balance the substitution
        return np.imag(phi_z * np.exp(-1j * u * t_linear * noise_w)) / v
        
    # integrating v from a tiny number to infinity
    val, _ = integrate.quad(gp_integrand, 1e-6, np.inf, epsabs=1e-3, epsrel=1e-3, limit=100)
    return 0.5 + (1.0 / np.pi) * val

# testing 15 thresholds so the s curve actually looks smooth
math_thresholds_db = np.linspace(-20, 20, 15)
math_coverage_probs = []

for t_db in math_thresholds_db:
    print(f"calculating for {t_db:.1f} db...")
    t_linear = 10**(t_db / 10.0)
    
    def outer_integral(r0):
        # integrating over all possible serving distances using the law of total probability
        prob_los = p_al(r0) * gil_pelaez_math(t_linear, r0, True)
        prob_nlos = p_an(r0) * gil_pelaez_math(t_linear, r0, False)
        return pdf_r0(r0) * (prob_los + prob_nlos)
        
    # bounding it to 1000 instead of sim_radius so it doesn't do a bunch of useless empty math
    final_prob, _ = integrate.quad(outer_integral, 0, 1000, epsabs=1e-3, epsrel=1e-3, limit=50)
    math_coverage_probs.append(final_prob)

print("done plotting the graphs now")

# setting up the superimposed graph
plt.figure(figsize=(9, 6))

# solid blue line for the monte carlo simulation
plt.plot(thresholds_db_sim, sim_coverage_probs, label='Simulation (10,000 drops)', color='blue', linewidth=2)

# red dashed line with markers for the math formulas
plt.plot(math_thresholds_db, math_coverage_probs, label='Analytical Math (Gil-Pelaez)', color='red', marker='^', linestyle='--')

plt.xlabel('Target SINR Threshold T (dB)')
plt.ylabel('Coverage Probability P_cov(T)')
plt.title('Validation of UAV Coverage Probability')
plt.legend()
plt.grid(True)
plt.xlim(-20, 20)
plt.ylim(0, 1.05)
plt.show()