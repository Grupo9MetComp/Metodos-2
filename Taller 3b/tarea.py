import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
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


@njit(fastmath=True)
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


# 2
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["figure.figsize"] = (14,5)
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["image.origin"] = "upper"
np.set_printoptions(linewidth=120)

import seaborn as sns

# 2
def crear(N_x, N_t, dx, dt, c):
    x = np.arange(0,dx*N_x,dx)
    k = ((c**2)*(dt**2))/(dx**2)
    U = np.zeros((N_t, N_x))*np.nan
    ## Condiciones iniciales
    for i in range(N_x):
        exp = (-125)*((x[i]-0.5)**2)
        U[0][i] = np.exp(exp)
        U[1][i] = np.exp(exp)
    return U, k, x

## Simulación
@njit
def evolve_dirichlet(U, k):
    for i in range(1, len(U)-1):
        for j in range(1, len(U[0])-1):
            U[i+1, j] = 2*(1-k)*U[i, j] + k*U[i, j+1] + k*U[i, j-1] - U[i-1, j]
        U[i+1][0] = 0
        U[i+1][-1] = 0
    return U

@njit
def evolve_neumann(U, k):
    for i in range(1, len(U)-1):
        for j in range(1, len(U[0])-1):
            U[i+1, j] = 2*(1-k)*U[i, j] + k*U[i, j+1] + k*U[i, j-1] - U[i-1, j]
        U[i+1][0] = U[i+1][1]
        U[i+1][-1] = U[i+1][-2]        
    return U

@njit
def evolve_periodicas(U, k):
    for i in range(1, len(U)-1):
        for j in range(len(U[0])):
            jp1 = (j+1)%len(U[0])
            jm1 = (j-1)%len(U[0])
            U[i+1, j] = 2*(1-k)*U[i, j] + k*U[i, jp1] + k*U[i, jm1] - U[i-1, j]
    return U
    

N_x = 10
N_t = 3
dx = 0.2
dt = 0.3
c = 1
U, k, x = crear(N_x, N_t, dx, dt, c)
evolve_dirichlet(U, k)
evolve_neumann(U, k)
evolve_periodicas(U, k)

N_x = 100
N_t = 300
dx = 0.02
dt = 0.01
c = 1
U, k, x = crear(N_x, N_t, dx, dt, c)
sol_dir = evolve_dirichlet(U, k)

U, k, x = crear(N_x, N_t, dx, dt, c)
sol_neu = evolve_neumann(U, k)

U, k, x = crear(N_x, N_t, dx, dt, c)
sol_per = evolve_periodicas(U, k)

fig, (ax1, ax2, ax3) = plt.subplots(ncols=1, nrows=3, frameon=False)
fig.subplots_adjust(hspace=1)
points_dir = ax1.scatter(x,sol_dir[0],c=sol_dir[0],cmap="coolwarm",s=500,vmin=-1, vmax=1)
ax1.set_title('Dirichlet')
ax1.set_facecolor('k')
ax1.set_xlim(0, 2)
ax1.set_ylim(-1.1, 1.1)
points_neu = ax2.scatter(x,sol_neu[0],c=sol_neu[0],cmap="coolwarm",s=500,vmin=-1, vmax=1)
ax2.set_title('Neumann')
ax2.set_facecolor('k')
ax2.set_xlim(0, 2)
ax2.set_ylim(0, 1.1)
points_per = ax3.scatter(x,sol_per[0],c=sol_per[0],cmap="coolwarm",s=500,vmin=-1, vmax=1)
ax3.set_title('Periodicas')
ax3.set_facecolor('k')
ax3.set_xlim(0, 2)
ax3.set_ylim(0, 1.1)


def draw_frame(frame):
    points_dir.set_offsets(np.transpose([x,sol_dir[frame]]))
    points_dir.set_array(sol_dir[frame]) # set colors
    points_neu.set_offsets(np.transpose([x,sol_neu[frame]]))
    points_neu.set_array(sol_neu[frame])
    points_per.set_offsets(np.transpose([x,sol_per[frame]]))
    points_per.set_array(sol_per[frame])
    return points_dir, points_neu, points_per

anim = animation.FuncAnimation(fig,draw_frame,frames=range(0,len(U)//2,12))
anim.save('2.mp4')