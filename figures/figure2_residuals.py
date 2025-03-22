import numpy as np
import matplotlib.pyplot as plt

# Cosmological parameters
H0 = 68.2  # km/s/Mpc
Omega_m = 0.310
k_stretch = 1.60e-18  # s^-1
n = 1.01

# Redshift range
z = np.linspace(0.01, 2.0, 100)

# Luminosity distance d_L(z) = (1+z) / H0 * integral_0^z dz' / E(z')
# E(z) = H(z) / H0
def E_stretch(z):
    H = np.zeros_like(z)
    H[0] = H0
    for i in range(1, len(z)):
        a = 1 / (1 + z[i])
        rho_m = Omega_m * H0**2 * (1 + z[i])**3
        H[i] = np.sqrt(rho_m + (k_stretch / 3) * H[i-1]**n / (8 * np.pi / 3))
    return H / H0

# Compute d_L(z) using numerical integration
def d_L(z, E):
    dz = z[1] - z[0]
    integral = np.cumsum(1 / E) * dz
    return (1 + z) * integral / H0  # in Mpc

# Mock Pantheon+ data
z_pantheon = np.linspace(0.1, 2.0, 20)
d_L_data = np.array([500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000,
                     5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500, 10000])  # Mock d_L in Mpc
d_L_err = d_L_data * 0.02  # 2% error

# Compute d_L for stretch model
E_vals = E_stretch(z)
d_L_model = d_L(z, E_vals)

# Interpolate model d_L to match Pantheon+ redshifts
d_L_model_interp = np.interp(z_pantheon, z, d_L_model)

# Compute residuals
residuals = (d_L_model_interp - d_L_data) / d_L_data * 100  # Percentage residuals

# Plotting
plt.figure(figsize=(8, 6))
plt.errorbar(z_pantheon, residuals, yerr=2.0, fmt='o', color='blue', label='Stretch Model')
plt.axhline(0, color='black', linestyle='--')
plt.xlabel('Redshift $z$')
plt.ylabel('$\Delta d_L / d_L \\times 100$ (%)')
plt.title('Percentage Residuals for Pantheon+ Data')
plt.legend()
plt.grid(True)
plt.savefig('figure2_residuals.png', dpi=300, bbox_inches='tight')
plt.close()
