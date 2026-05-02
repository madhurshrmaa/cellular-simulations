# Cellular Networks Stochastic Geometry Simulations

This repository contains Python simulations modeling large-scale cellular networks using stochastic geometry. 

The codebase implements mathematical models and validates theoretical coverage probabilities presented in two foundational papers:
1. *Modeling and Analysis of Cellular Networks Using Stochastic Geometry: A Tutorial* (ElSawy et al., 2017).
2. *A Tractable Approach to Coverage and Rate in Cellular Networks* (Andrews, Baccelli, and Ganti, 2011).

## Current State of the Simulations

The repository has advanced past the baseline setup. The current codebase successfully models a complete interference-limited network and implements the following features:

* **Poisson Point Process Rendering (PPP):** Models the spatial distribution of Base Stations (BS) using a Homogeneous Poisson Point Process (PPP) and maps the interference exclusion region using nearest-BS association.
* **Network Interference & Fading:** Calculates the exact Signal-to-Interference Noise Ratio (SINR) for a test user by simulating Rayleigh fading (exponential distribution) and utilizing a path-loss exponent of α = 4. The network is assumed to be interference-limited, meaning thermal noise is ignored.
* **Monte Carlo Validation:** Runs 10,000 randomized network snapshots to generate a Probability of Coverage curve (CCDF).
* **Theoretical Formula Comparison:** Directly compares the simulated 10,000-run network data against the closed-form mathematical model for coverage probability (Equation 14) derived by Andrews et al.

## How to Run
1. Clone the repository in your local hard drive: `git clone https://github.com/madhurshrmaa/cellular-simulations.git`
2. Activate the virtual environment. `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the required scripts. e.g.: `python3 compare_results.py`
