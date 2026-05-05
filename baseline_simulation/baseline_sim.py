import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial.distance import cdist

def simulate_baseline_system_model(lambda_intensity, area_size):
    """
    Simulates Section III of "Modeling and Analysis of Cellular Networks Using Stochastic Geometry"
    Generates a PPP of Base Stations, forms Voronoi cells, and maps the exclusion region (r_0).
    """
    
    # The area is a square from -area_size/2 to area_size/2 (400x400 sq units)
    total_area = area_size ** 2
    
    # The number of points in bounded region is a Poisson random variable.
    expected_num_bs = lambda_intensity * total_area
    num_bs = np.random.poisson(expected_num_bs)
    
    # Ψ = {xᵢ} represents the coordinates of the i-th BS.
    x_coords = np.random.uniform(-area_size/2, area_size/2, num_bs)
    y_coords = np.random.uniform(-area_size/2, area_size/2, num_bs)
    Psi_locations = np.column_stack((x_coords, y_coords))
    

    # The analysis is conducted for a test user located at the origin.
    origin_user = np.array([[0, 0]])
    
    # Calculate all distances from the test user to all BSs.
    # The set consists of the ordered BSs distances to the test user.
    distances = cdist(origin_user, Psi_locations)[0]
    
    # The user is associated with the closest BS.
    serving_bs_index = np.argmin(distances)
    
    # r₀ is the serving distance
    r_0 = distances[serving_bs_index]
    x_0 = Psi_locations[serving_bs_index] # Location of the serving BS
    
    # The set Ψ̃ \ r₀ represents the interfering BSs.
    interfering_bs_indices = [i for i in range(num_bs) if i != serving_bs_index]
    interfering_bs_locations = Psi_locations[interfering_bs_indices]
    
    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10), facecolor='#f4f4f9')
    ax.set_facecolor('#f4f4f9')
    
    # 1. Plotting the Voronoi-tessellation representing the service area of each BS.
    vor = Voronoi(Psi_locations)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, 
                    line_colors='dimgrey', line_width=1.2, line_alpha=0.7, line_style='-')
    
    # 2. Plotting the Interfering Base Stations 
    ax.scatter(interfering_bs_locations[:, 0], interfering_bs_locations[:, 1], 
               c='darkorange', marker='s', s=35, label='Interfering BSs ($\Psi \setminus x_0$)')
    
    # 3. Plotting the Serving Base Station (x₀)
    ax.scatter(x_0[0], x_0[1], c='purple', marker='s', s=80, 
               edgecolors='black', zorder=5, label='Serving BS ($x_0$)')
    
    # 4. Plotting the Test User at the Origin
    ax.scatter(0, 0, c='crimson', marker='*', s=150, 
               edgecolors='black', zorder=5, label='Test User (Origin)')
    
    # 5. Plotting the Interference Exclusion Region (circle of radius r₀)

    # Visualization of the spatial interference protection around the receiver
    exclusion_circle = plt.Circle((0, 0), r_0, color='crimson', fill=True, 
                                  alpha=0.15, linestyle='--', linewidth=2,
                                  label=f'Exclusion Region ($r_0 = {r_0:.2f}$)')
    exclusion_border = plt.Circle((0, 0), r_0, color='crimson', fill=False, 
                                  linestyle='--', linewidth=2)
    ax.add_patch(exclusion_circle)
    ax.add_patch(exclusion_border)
    
    # Formatting to make it clean and distinct
    ax.set_xlim(-area_size/2, area_size/2)
    ax.set_ylim(-area_size/2, area_size/2)
    ax.set_aspect('equal')
    ax.set_title(f"Baseline Cellular Network Model via SG\nIntensity ($\lambda$) = {lambda_intensity} BS/unit$^2$ | Total BSs = {num_bs}", 
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.9, fontsize=10)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.show()

# Runing the Simulation
# Setting a modest intensity to ensure the polygons and exclusion zone are highly visible
simulate_baseline_system_model(lambda_intensity=0.0005, area_size=500)