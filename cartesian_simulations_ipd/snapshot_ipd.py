import numpy as np
import matplotlib.pyplot as plt

# setup variables
lambda_bs = 0.000005 # density (towers per sq meter)
sim_radius = 5000 # radius in meters
alpha = 4.0 # path loss exponent
pt = 1.0 # transmit power in watts

# 1. generate number of BS in the big square
sim_area = (2 * sim_radius) ** 2
num_bs = np.random.poisson(lambda_bs * sim_area)

# 2. generate cartesian coords
x_coords_initial = np.random.uniform(-sim_radius, sim_radius, num_bs)
y_coords_initial = np.random.uniform(-sim_radius, sim_radius, num_bs)

distances_from_origin = np.sqrt(x_coords_initial**2 + y_coords_initial**2)

# 3. crop to circle to prevent edge effects
valid_mask = (distances_from_origin <= sim_radius)
distances = distances_from_origin[valid_mask]
x_coords = x_coords_initial[valid_mask]
y_coords = y_coords_initial[valid_mask]

# 4. bounded path loss from thesis: min(1, r^-alpha)
# this prevents the math from exploding if a tower spawns at r=0
path_loss = np.minimum(1.0, distances**-alpha)

# 5. rayleigh fading
fading_channels = np.random.exponential(1.0, size=len(distances))

# 6. calc IPD (sum of everything, no serving vs interfering distinction anymore)
received_powers = pt * fading_channels * path_loss
total_ipd_watts = np.sum(received_powers)

print("--- SNAPSHOT RESULTS ---")
print(f"Total Towers in range: {len(distances)}")

# BUG FIX: use scientific notation because watts at this distance are tiny
print(f"Calculated IPD Exposure (Watts): {total_ipd_watts:.2e} W")

# convert to dBm (dBm is used in appendix a)
# 1 Watt = 30 dBm. Formula: 10 * log10(watts) + 30
total_ipd_dbm = 10 * np.log10(total_ipd_watts) + 30
print(f"Calculated IPD (dBm): {total_ipd_dbm:.2f} dBm")

# 7. plot the map
fig, ax = plt.subplots(figsize=(8, 8))

# plot all base stations (all are just contributors now)
ax.scatter(x_coords, y_coords, c='orange', s=20, label='Base Stations')
# plot user in center
ax.scatter(0, 0, c='red', s=100, marker='*', label='You (User)')

ax.set_aspect('equal')
ax.set_xlim(-sim_radius, sim_radius)
ax.set_ylim(-sim_radius, sim_radius)
ax.legend()
plt.title(f"Network Snapshot\nTotal Exposure: {total_ipd_dbm:.2f} dBm")
plt.show()