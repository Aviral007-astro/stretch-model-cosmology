import numpy as np
import matplotlib.pyplot as plt

# Cosmological parameters
H0 = 68.2
Omega_m = 0.310
k_stretch = 1.60e-18
n = 1.01

# Redshift range
z = np.linspace(0, 2, 100)

# H(z) for stretch model
def H_stretch(z):
    H = np.zeros_like(z)
    H[0] = H0
    for i in range(1, len(z)):
        a = 1 / (1 + z[i])
        rho_m = Omega_m * H0**2 * (1 + z[i])**3
        H[i] = np.sqrt(rho_m + (k_stretch / 3) * H[i-1]**n / (8 * np.pi / 3))
    return H

# Compute H(z) and its derivative
H_vals = H_stretch(z)
dz = z[1] - z[0]
dHdz = np.gradient(H_vals, dz)

# Equation of state: w = -1 - (1/3) * (dH/dz) / H
w_stretch = -1 - (1/3) * dHdz / H_vals

# ΛCDM: w = -1
w_lcdm = -1 * np.ones_like(z)

# CPL parameterization: w(z) = w0 + wa * z / (1+z)
w0 = -0.936
wa = -0.314
w_cpl = w0 + wa * z / (1 + z)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(z, w_stretch, label='Stretch Model', color='blue')
plt.plot(z, w_lcdm, label='ΛCDM ($w = -1$)', color='red', linestyle='--')
plt.plot(z, w_cpl, label='CPL ($w_0 = -0.936, w_a = -0.314$)', color='green', linestyle='-.')
plt.xlabel('Redshift $z$')
plt.ylabel('$w_{\\text{stretch}}(z)$')
plt.title('Equation of State $w_{\\text{stretch}}(z)$')
plt.legend()
plt.grid(True)
plt.savefig('figure5_wz.png', dpi=300, bbox_inches='tight')
plt.close()
