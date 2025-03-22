import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# Cosmological parameters
H0 = 68.2
Omega_m = 0.310
k_stretch = 1.60e-18
n = 1.01
H0_lcdm = 67.4
Omega_m_lcdm = 0.315

# Redshift range
z = np.linspace(0, 10, 100)

# H(z) for stretch model
def H_stretch(z):
    H = np.zeros_like(z)
    H[0] = H0
    for i in range(1, len(z)):
        a = 1 / (1 + z[i])
        rho_m = Omega_m * H0**2 * (1 + z[i])**3
        H[i] = np.sqrt(rho_m + (k_stretch / 3) * H[i-1]**n / (8 * np.pi / 3))
    return H

# H(z) for ΛCDM
def H_lcdm(z):
    Omega_L = 1 - Omega_m_lcdm
    return H0_lcdm * np.sqrt(Omega_m_lcdm * (1 + z)**3 + Omega_L)

# Growth factor D(z) by solving the differential equation: d^2δ/dz^2 + ...
# Simplified: Solve dδ/dz + (2 + dlnH/dz) δ = (3/2) Ω_m δ
def growth_eq(y, z, H_func):
    delta, delta_prime = y
    H = H_func(z)
    dHdz = np.gradient(H, z[1] - z[0])
    dlnHdz = dHdz / H
    Omega_m_z = Omega_m * (1 + z)**3 * (H0 / H)**2
    d2delta_dz2 = -(2 + dlnHdz) * delta_prime + (3/2) * Omega_m_z * delta
    return [delta_prime, d2delta_dz2]

# Solve for stretch model
H_stretch_vals = H_stretch(z)
sol_stretch = odeint(growth_eq, [1.0, 0.0], z, args=(H_stretch,))
D_stretch = sol_stretch[:, 0] / sol_stretch[0, 0]  # Normalize to D(z=0) = 1

# Solve for ΛCDM
H_lcdm_vals = H_lcdm(z)
sol_lcdm = odeint(growth_eq, [1.0, 0.0], z, args=(H_lcdm,))
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[0, 0]

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(z, D_stretch, label='Stretch Model', color='blue')
plt.plot(z, D_lcdm, label='ΛCDM', color='red', linestyle='--')
plt.xlabel('Redshift $z$')
plt.ylabel('$D(z)$')
plt.title('Growth Factor $D(z)$')
plt.legend()
plt.grid(True)
plt.savefig('figure8_Dz.png', dpi=300, bbox_inches='tight')
plt.close()
