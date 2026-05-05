import numpy as np
import matplotlib.pyplot as plt

# setup variables
base_station_density = 0.000005 # made up number (0.000005 stations per m^2)
simulation_radius = 5000 # simulate infinity by making the radius bigger (in m) 
path_loss_exponent = 4.0 # paper assumes this value by ignoring noise

# 1. generate number of BS in the big square
# L = 2 * radius, so Area = L^2
simulation_area_square = (2 * simulation_radius) ** 2
num_base_stations = np.random.poisson(base_station_density * simulation_area_square) # expected numbers of stations

# 2. generate cartesian coords
# generate x and y uniformly between -simulation_radius and +simulation_radius
x_coordinates_initial = np.random.uniform(-simulation_radius, simulation_radius, num_base_stations)
y_coordinates_initial = np.random.uniform(-simulation_radius, simulation_radius, num_base_stations)

# calc distance from origin using pythagorean theorem
distances_from_origin = np.sqrt(x_coordinates_initial**2 + y_coordinates_initial**2)

# 3. apply rejection sampling (crop to circle) and 1m exclusion zone
# keep points inside the circle (<= simulation_radius) AND outside the exclusion zone (>= 1.0)
valid_mask = (distances_from_origin <= simulation_radius) & (distances_from_origin >= 1.0)

distances_from_origin = distances_from_origin[valid_mask]
x_coordinates = x_coordinates_initial[valid_mask]
y_coordinates = y_coordinates_initial[valid_mask]

# 4. find serving BS (the closest one)
sorted_indices = np.argsort(distances_from_origin) # sort arrays of distances from smallest to largest
sorted_distances = distances_from_origin[sorted_indices]
sorted_x = x_coordinates[sorted_indices]
sorted_y = y_coordinates[sorted_indices]

distance_to_serving_bs = sorted_distances[0]

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

# 7. plot the map (coordinates are already in xy)
fig, ax = plt.subplots(figsize=(8, 8))

# plot interferers
ax.scatter(sorted_x[1:], sorted_y[1:], c='orange', s=20, label='Interfering BS')
# plot serving
ax.scatter(sorted_x[0], sorted_y[0], c='purple', s=80, marker='s', label='Serving BS')
# plot user
ax.scatter(0, 0, c='red', s=100, marker='*', label='Test User')

# draw the 1m exclusion zone (might be too small to see on a 5000m map, but zooming in can help)
exclusion_zone_circle = plt.Circle((0,0), 1.0, color='red', fill=False, linestyle='--')
ax.add_patch(exclusion_zone_circle)

ax.set_aspect('equal')
ax.set_xlim(-simulation_radius, simulation_radius)
ax.set_ylim(-simulation_radius, simulation_radius)
ax.legend()
plt.title(f"One Network Snapshot (Cartesian Generated)\nSINR: {10*np.log10(signal_to_interference_ratio):.2f} dB")
plt.show()