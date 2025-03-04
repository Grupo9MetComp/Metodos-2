import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba
from mpl_toolkits.mplot3d import Axes3D

# 1

N = 500
L = 1.1
diff = 2*L/(N-1)
x_vals = np.linspace(-L, L, N)
y_vals = np.linspace(-L, L, N)
X, Y = np.meshgrid(x_vals, y_vals, indexing='xy')

dentro = X**2 + Y**2 < 1  
frontera = np.logical_not(dentro)

theta = np.arctan2(Y, X)
phi = np.random.rand(N, N) * dentro  
phi[frontera] = np.sin(7 * theta[frontera])

phi_initial = phi.copy()
fun = -4*np.pi*(-X-Y)*dentro 


@numba.njit(fastmath=True)
def solve_poisson(phi, fun, dentro, diff, max_iter=15000, tol=1e-4):
    N = phi.shape[0]
    for k in range(max_iter):
        phi_new = phi.copy()
        for i in range(1, N-1):
            for j in range(1, N-1):
                if dentro[i, j]:  
                    phi_new[i, j] = 0.25 * (phi[i+1, j] + phi[i-1, j] +phi[i, j+1] + phi[i, j-1] + diff**2* fun[i, j])
        if np.trace(np.abs(phi_new - phi)) < tol:
            break
        phi[:] = phi_new
    return phi

phi = solve_poisson(phi, fun , dentro, diff)
fig1, axes = plt.subplots(1, 2, figsize=(14, 5))

im = axes[0].pcolormesh(X, Y, phi_initial, cmap="jet", shading='auto')
fig1.colorbar(im, ax=axes[0], label="φ")
axes[0].set_title("Condiciones iniciales")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
ax = fig1.add_subplot(1, 2, 2, projection='3d')
ax.plot_surface(X, Y, phi, cmap="jet", shade=True)
ax.set_title("Solución")
plt.savefig("1.png", dpi=350)
