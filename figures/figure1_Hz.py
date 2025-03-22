import numpy as np
import matplotlib.pyplot as plt

# Cosmological parameters
H0 = 68.2  # Hubble constant (km/s/Mpc) for stretch model
Omega_m = 0.310  # Matter density parameter
k_stretch = 1.60e-18  # Stretch constant (s^-1)
n = 1.01  # Stretch exponent
H0_lcdm = 67.4  # Hubble constant for ΛCDM (km/s/Mpc)
Omega_m_lcdm = 0.315  # Matter density for ΛCDM

# Redshift range
z = np.linspace(0, 2.5, 100)

# Stretch model: H(z) = sqrt(8πG/3 ρ_m + k_stretch H^n / 3)
# For simplicity, we solve iteratively for H(z) assuming n ≈ 1
def H_stretch(z):
    H = np.zeros_like(z)
    H[0] = H0
    for i in range(1, len(z)):
        a = 1 / (1 + z[i])
        rho_m = Omega_m * H0**2 * (1 + z[i])**3  # Matter density
        H[i] = np.sqrt(rho_m + (k_stretch / 3) * H[i-1]**n / (8 * np.pi / 3))
    return H

# ΛCDM model: H(z) = H0 * sqrt(Ω_m (1+z)^3 + Ω_Λ)
def H_lcdm(z):
    Omega_L = 1 - Omega_m_lcdm
    return H0_lcdm * np.sqrt(Omega_m_lcdm * (1 + z)**3 + Omega_L)

# Mock DESI DR2 data (H(z) at specific redshifts with error bars)
z_desi = np.array([0.5, 1.0, 1.5, 2.0])
H_desi = np.array([83.0, 95.0, 105.0, 120.0])  # Mock H(z) values
H_desi_err = np.array([5.0, 6.0, 7.0, 8.0])  # Mock error bars

# Compute H(z) for both models
H_stretch_vals = H_stretch(z)
H_lcdm_vals = H_lcdm(z)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(z, H_stretch_vals, label='Stretch Model', color='blue')
plt.plot(z, H_lcdm_vals, label='ΛCDM', color='red', linestyle='--')
plt.errorbar(z_desi, H_desi, yerr=H_desi_err, fmt='o', color='black', label='DESI DR2 Data')
plt.xlabel('Redshift $z$')
plt.ylabel('$H(z)$ (km/s/Mpc)')
plt.title('$H(z)$ vs. Redshift')
plt.legend()
plt.grid(True)
plt.savefig('figure1_Hz.png', dpi=300, bbox_inches='tight')
plt.close()
