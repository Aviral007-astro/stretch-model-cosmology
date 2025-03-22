import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.interpolate import interp1d

# Cosmological parameters
H0 = 68.2  # km/s/Mpc
H0_s = H0 * 3.24078e-20  # s^-1
Omega_m = 0.310
k_stretch = 1.60e-18  # s^-1
G = 4.302e-9  # Mpc/Msun (km/s)^2

z = np.linspace(0, 2, 1000)
a = 1 / (1 + z)

def solve_E(z):
    a = 1
    b = -k_stretch / (3 * H0_s)
    c = -Omega_m * (1 + z)**3
    return (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

E_stretch = np.array([solve_E(zi) for zi in z])
H_stretch = H0 * E_stretch
E_lcdm = np.sqrt(Omega_m * (1 + z)**3 + 1 - Omega_m)
H_lcdm = H0 * E_lcdm

def dH_da(H, a, z):
    dz_da = -(1 + z)**2
    dH_dz = np.gradient(H, z)
    return dH_dz * dz_da

dH_stretch = dH_da(H_stretch, a, z)
dH_lcdm = dH_da(H_lcdm, a, z)
H_interp_stretch = interp1d(a[::-1], H_stretch[::-1], fill_value="extrapolate")
dH_interp_stretch = interp1d(a[::-1], dH_stretch[::-1], fill_value="extrapolate")
H_interp_lcdm = interp1d(a[::-1], H_lcdm[::-1], fill_value="extrapolate")
dH_interp_lcdm = interp1d(a[::-1], dH_lcdm[::-1], fill_value="extrapolate")

def growth_eq(y, a, H_interp, dH_interp):
    delta, ddelta = y
    H = H_interp(a)
    dH = dH_interp(a)
    Om_a = Omega_m / (a**3 * (H / H0)**2)
    d2 = -(3/a + dH/H) * ddelta + 1.5 * Om_a * delta / a**2
    return [ddelta, d2]

from scipy.integrate import odeint
sol_stretch = odeint(growth_eq, [1.0, 0.0], a, args=(H_interp_stretch, dH_interp_stretch))
sol_lcdm = odeint(growth_eq, [1.0, 0.0], a, args=(H_interp_lcdm, dH_interp_lcdm))

D_stretch = sol_stretch[:, 0] / sol_stretch[-1, 0]
D_lcdm = sol_lcdm[:, 0] / sol_lcdm[-1, 0]

plt.plot(z, D_stretch, label="Stretch Model", color="blue")
plt.plot(z, D_lcdm, label="ΛCDM", color="red", linestyle="--")
plt.xlabel("Redshift z")
plt.ylabel("Growth Factor D(z)")
plt.title("Growth Factor vs Redshift")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("figure8_Dz.png", dpi=300)
