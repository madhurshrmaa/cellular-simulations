import numpy as np
import matplotlib.pyplot as plt

# x axis thresholds from -10 to 20 dB (like the paper's graph)
target_sinr_thresholds_db = np.linspace(-10, 20, 100) # linear space of 100 evenly spaced numbers

# convert dB to linear for the math formula
target_sinr_thresholds_linear = 10 ** (target_sinr_thresholds_db / 10.0)

# Eq 14 from the paper (alpha=4, no noise, exponential fading)
# Formula: 1 / (1 + sqrt(T) * (pi/2 - arctan(1/sqrt(T))))
sqrt_of_linear_threshold = np.sqrt(target_sinr_thresholds_linear)
arctan_term = (np.pi / 2.0) - np.arctan(1.0 / np.sqrt(target_sinr_thresholds_linear))
theoretical_coverage_probability = 1.0 / (1.0 + sqrt_of_linear_threshold * arctan_term)

plt.figure(figsize=(8, 6))
# triangles
plt.plot(target_sinr_thresholds_db, theoretical_coverage_probability, label='Math Formula (Eqn 14)', color='red', marker='^', markevery=5, linestyle='--') ## triangle on every fifth data point

plt.xlim(-10, 20)
plt.ylim(0, 1)
plt.xlabel('Target SINR Threshold (dB)')
plt.ylabel('Probability of Coverage')
plt.title('Formulaic Math Result (No H-PPP)')
plt.legend()
plt.grid(True)
plt.show()