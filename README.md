# Cellular Networks Stochastic Geometry Simulations

This repository contains Python simulations based on the mathematical models presented in *Modeling and Analysis of Cellular Networks Using Stochastic Geometry: A Tutorial* (ElSawy et al.).

## Disclaimer
Note: This is currently just the initial setup and baseline spatial abstraction (Section III). The current code models the Poisson Point Process (PPP), nearest-BS association, and maps the interference exclusion region ($r_0$). More advanced simulations covering aggregate interference ($i_{agg}$) and SINR outage probabilities are actively being developed and will be pushed soon.

## How to Run
1. Clone the repository in your local hard drive: `git clone https://github.com/madhurshrmaa/cellular-simulations.git`
2. Activate the virtual environment. `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt`
4. Run the baseline script: `python3 baseline_sim.py`
