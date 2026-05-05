import numpy as np
import matplotlib.pyplot as plt

# 1.SIMULATION (10,000 runs)
number_of_simulations = 10000 # run 10k times
base_station_density = 0.0001
simulation_radius = 10000 # larger radius to prevent edge effects
path_loss_exponent = 4.0

simulated_sinr_db_values = [] # container to collect all SINR values in db
print("simulating 10,000 networks... please wait...")

# drop all towers, apply 1-m exclusion zone, find closest tower, apply fading_serving_channel, divide signal by interference
for _ in range(number_of_simulations):
    # L = 2 * radius, so Area = L^2
    simulation_area_square = (2 * simulation_radius) ** 2
    num_base_stations = np.random.poisson(base_station_density * simulation_area_square)
    if num_base_stations == 0: continue # if 0 towers spawn, skip this iteration
    
    # generate x and y uniformly between -simulation_radius and +simulation_radius
    x_coordinates_initial = np.random.uniform(-simulation_radius, simulation_radius, num_base_stations)
    y_coordinates_initial = np.random.uniform(-simulation_radius, simulation_radius, num_base_stations)
    
    # calc distance from origin using pythagorean theorem
    distances_from_origin = np.sqrt(x_coordinates_initial**2 + y_coordinates_initial**2)
    
    # apply rejection sampling (crop to circle) and 1m exclusion zone
    # keep points inside the circle (<= simulation_radius) AND outside the exclusion zone (>= 1.0)
    valid_mask = (distances_from_origin <= simulation_radius) & (distances_from_origin >= 1.0)
    distances_from_origin = distances_from_origin[valid_mask]
    
    if len(distances_from_origin) == 0: continue # if all bs were filtered out, skip this iteration
    
    distances_from_origin = np.sort(distances_from_origin)
    distance_to_serving_bs = distances_from_origin[0]
    distances_to_interfering_bs = distances_from_origin[1:]
    
    fading_serving_channel = np.random.exponential(1.0)
    fading_interfering_channels = np.random.exponential(1.0, size=len(distances_to_interfering_bs))
    
    received_signal_power = fading_serving_channel * (distance_to_serving_bs ** -path_loss_exponent)
    total_interference_power = np.sum(fading_interfering_channels * (distances_to_interfering_bs ** -path_loss_exponent))
    
    if total_interference_power > 0: # if there's interference at all
        signal_to_interference_ratio = received_signal_power / total_interference_power
        simulated_sinr_db_values.append(10 * np.log10(signal_to_interference_ratio)) # take all values of SINR, convert to db, append in simulated_sinr_db_values

# turn the 10k results into a probability curve
simulated_sinr_db_values = np.sort(simulated_sinr_db_values)
# probability = 100% minus the current index divided by total [CCDF: P(SINR > x)]
simulated_coverage_probability = 1.0 - np.arange(1, len(simulated_sinr_db_values) + 1) / len(simulated_sinr_db_values) # average 

# 2. THE MATH / FORMULA ONLY (NO H-PPP)
target_sinr_thresholds_db = np.linspace(-10, 20, 100)
target_sinr_thresholds_linear = 10 ** (target_sinr_thresholds_db / 10.0)
theoretical_coverage_probability = 1.0 / (1.0 + np.sqrt(target_sinr_thresholds_linear) * ((np.pi / 2.0) - np.arctan(1.0 / np.sqrt(target_sinr_thresholds_linear))))

# 3. COMPARE THE TWO 
plt.figure(figsize=(9, 6))

# blue solid line for our 10k simulations
plt.plot(simulated_sinr_db_values, simulated_coverage_probability, label='Simulation (10,000 random PPPs)', color='blue')
# red dashed line with triangles for the paper's math
plt.plot(target_sinr_thresholds_db, theoretical_coverage_probability, label='Math Formula', color='red', marker='^', markevery=5, linestyle='--')

plt.xlim(-10, 20)
plt.ylim(0, 1)
plt.xlabel('Target SINR Threshold (dB)')
plt.ylabel('Probability of Coverage (CCDF)')
plt.title('My simulation vs the math in the paper')
plt.legend()
plt.grid(True)
plt.show()