import numpy as np
import matplotlib.pyplot as plt
import corner
import emcee

# Mock MCMC data (simulating the posterior distributions)
# True values from the paper
n_true = 1.01
k_stretch_true = 1.60e-18
Omega_m_true = 0.310
H0_true = 68.2

# Generate mock MCMC samples
np.random.seed(42)
n_samples = 10000
n = np.random.normal(n_true, 0.04, n_samples)  # n = 1.01 ± 0.04
k_stretch = np.random.normal(k_stretch_true, 0.06e-18, n_samples)  # k_stretch = (1.60 ± 0.06) × 10^-18
Omega_m = np.random.normal(Omega_m_true, 0.006, n_samples)  # Omega_m = 0.310 ± 0.006
H0 = np.random.normal(H0_true, 0.8, n_samples)  # H0 = 68.2 ± 0.8

# Combine samples into a single array
samples = np.vstack((n, k_stretch, Omega_m, H0)).T

# Labels for the corner plot
labels = ['$n$', '$k_{\\text{stretch}}$ (s$^{-1}$)', '$\\Omega_m$', '$H_0$ (km/s/Mpc)']

# Create the corner plot
fig = corner.corner(samples, labels=labels, truths=[n_true, k_stretch_true, Omega_m_true, H0_true],
                    quantiles=[0.16, 0.5, 0.84], show_titles=True, title_kwargs={"fontsize": 12})
plt.savefig('figure3_corner.png', dpi=300, bbox_inches='tight')
plt.close()
