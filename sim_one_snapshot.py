import numpy as np
import matplotlib.pyplot as plt

# setup variables
base_station_density = 0.000005 # made up number (0.000005 stations per m^2)
simulation_radius = 5000 # simulate infinity by making the radius bigger (in m) 
path_loss_exponent = 4.0 # paper assumes this value by ignoring noise

# 1. generate number of BS in the big circle
simulation_area = np.pi * (simulation_radius ** 2)
num_base_stations = np.random.poisson(base_station_density * simulation_area) # expected numbers of stations

# 2. generate polar coords
# sqrt(uniform) ensures they are evenly spread in the circle, not bunched in the middle
uniform_random_variables = np.random.uniform(0, 1, num_base_stations)

# probability a point lies within radius r should equal the fraction of total area inside that radius.
# area of bigger circle: πR^2, area upto radius r: πr^2; so P(R ≤ r)= r^2 / R^2
# so, uniform_random_variables = r^2 / R^2 => r = R sqrt(uniform_random_variables)
# sqrt(uniform_random_variables) converts a uniform random variable into one that accounts for the fact that circle area grows with r^2. 
# This ensures points are evenly distributed over the area, rather than clustered near the center.
# Source: LeetCode 478 Python
distances_from_origin = simulation_radius * np.sqrt(uniform_random_variables)
angles_in_radians = np.random.uniform(0, 2*np.pi, num_base_stations) # pick random fair angle between 0 and 2pi for every station

# 3. apply 1m exclusion zone
valid_distances_mask = distances_from_origin >= 1.0
distances_from_origin = distances_from_origin[valid_distances_mask]
angles_in_radians = angles_in_radians[valid_distances_mask]

# 4. find serving BS (the closest one)
sorted_indices = np.argsort(distances_from_origin) # sort arrays of distances from smallest to largest
sorted_distances = distances_from_origin[sorted_indices]
sorted_angles = angles_in_radians[sorted_indices]

distance_to_serving_bs = sorted_distances[0]
angle_to_serving_bs = sorted_angles[0]

# everything else is interference
distances_to_interfering_bs = sorted_distances[1:]

# 5. apply rayleigh fading (random exponential multiplier)
fading_serving_channel = np.random.exponential(1.0)
fading_interfering_channels = np.random.exponential(1.0, size=len(distances_to_interfering_bs)) # for every single multiplier

# 6. calc SINR (S / I, assuming no thermal noise)
received_signal_power = fading_serving_channel * (distance_to_serving_bs ** -path_loss_exponent)
total_interference_power = np.sum(fading_interfering_channels * (distances_to_interfering_bs ** -path_loss_exponent))
signal_to_interference_ratio = received_signal_power / total_interference_power

print(f"PRINTING SNAPSHOT RESULTS")
print(f"Serving BS Distance: {distance_to_serving_bs:.2f} meters")
print(f"Signal Power (S): {received_signal_power:.6f}")
print(f"Interference (I): {total_interference_power:.6f}")
print(f"Calculated SINR: {signal_to_interference_ratio:.4f}")
print(f"Calculated SINR (dB): {10 * np.log10(signal_to_interference_ratio):.2f} dB")

# 7. plot the map (convert polar to xy just for the scatter plot)
x_coordinates = sorted_distances * np.cos(sorted_angles)
y_coordinates = sorted_distances * np.sin(sorted_angles)

fig, ax = plt.subplots(figsize=(8, 8))

# plot interferers
ax.scatter(x_coordinates[1:], y_coordinates[1:], c='orange', s=20, label='Interfering BS')
# plot serving
ax.scatter(x_coordinates[0], y_coordinates[0], c='purple', s=80, marker='s', label='Serving BS')
# plot user
ax.scatter(0, 0, c='red', s=100, marker='*', label='Test User')

# draw the 1m exclusion zone (might be too small to see on a 5000m map, but zooming in can help)
exclusion_zone_circle = plt.Circle((0,0), 1.0, color='red', fill=False, linestyle='--')
ax.add_patch(exclusion_zone_circle)

ax.set_aspect('equal')
ax.set_xlim(-simulation_radius, simulation_radius)
ax.set_ylim(-simulation_radius, simulation_radius)
ax.legend()
plt.title(f"One Network Snapshot (Polar Generated)\nSINR: {10*np.log10(signal_to_interference_ratio):.2f} dB")
plt.show()