import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import numba
from scipy.integrate import solve_ivp
import matplotlib.animation as animation
from scipy.signal import find_peaks
from scipy.optimize import curve_fit

plt.style.use('dark_background')
plt.rcParams["animation.html"] = "jshtml"

g = 9.773
m = 10
v_0 = 10
dt = 0.001

@numba.njit(fastmath=True)
def derivs(vx, vy, beta):
    v = np.sqrt(vx**2 + vy**2)
    ax = -beta * vx * v
    ay = -g - beta * vy * v
    return ax, ay

@numba.njit(fastmath=True)
def parab_verlet(angle_grados, beta):
    angle_rad = np.radians(angle_grados)
    vx = v_0 * np.cos(angle_rad)
    vy = v_0 * np.sin(angle_rad)
    x, y = 0.0, 0.0
    E0 = 0.5 * m * v_0**2 + m * g * y  
    N_max = 10000
    xvals = np.zeros(N_max)
    yvals = np.zeros(N_max)
    ax, ay = derivs(vx, vy, beta)

    i = 0
    while y >= 0 and i < N_max:
        x += vx * dt + 0.5 * ax * dt**2
        y += vy * dt + 0.5 * ay * dt**2
        
        ax_new, ay_new = derivs(vx, vy, beta)

        vx += 0.5 * (ax + ax_new) * dt
        vy += 0.5 * (ay + ay_new) * dt

        xvals[i] = x
        yvals[i] = y

        ax, ay = ax_new, ay_new
        i += 1

    xvals = xvals[:i]
    yvals = yvals[:i]

    v_f = np.sqrt(vx**2 + vy**2)

    x_max = np.max(xvals)

    Ef = 0.5 * m * v_f**2 + m * g * yvals[-1]

    delta_e = Ef - E0

    return xvals, yvals, delta_e, -x_max


beta_test = 0
angle_test = 45  

x_vals, y_vals, energia_perdida, alcance = parab_verlet(angle_test, beta_test)
#print(f"Energía perdida con beta = {beta_test}: {energia_perdida:.6f} J")
#print(f"Alcance con beta = {beta_test}: {-alcance:.6f} m")
beta_values = np.logspace(np.log10(0.01), np.log10(2), 20)
beta_values = np.insert(beta_values, 0, 0)
theta_max_values = []
delta_e_vals = []


def fun_opt(angle_grados, beta):
    return parab_verlet(angle_grados, beta)[-1]

for beta in beta_values:
    res = sp.optimize.minimize_scalar(fun_opt, bounds=(0, 85), args=(beta), method="bounded")
    theta_max_values.append(res.x)
for beta in beta_values:
    delta_e_vals.append(-1*parab_verlet(45, beta)[2])
    

fig1a = plt.figure(figsize=(8,5))
plt.plot(beta_values, theta_max_values,'.-',c='b')
plt.xscale("log") 
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\theta_{\text{max}}$')
plt.grid('True')
plt.savefig('1 - a.pdf')

fig1b = plt.figure(figsize=(8,5))
plt.plot(beta_values, delta_e_vals,'.-',c='r')
plt.xscale("log")
plt.xlabel(r'$\beta$')
plt.ylabel(r'$\Delta E$')
plt.grid('True')
plt.savefig('1 - b.pdf')

# 2
def F(t, Y):
    x, y, vx, vy = Y
    r = np.array([x, y])
    norm_r = np.linalg.norm(r)
    a = -r/(norm_r**3)
    return np.array([vx, vy, a[0], a[1]])

def RK4_step(F,y0,t,dt):
    k1 = F(t,y0)
    k2 = F( t+dt/2, y0 + dt*k1/2 )
    k3 = F( t+dt/2, y0 + dt*k2/2  )
    k4 = F( t+dt, y0 + dt*k3  )
    return y0 + dt/6 * (k1+2*k2+2*k3+k4)

dt = 0.1
t_max = 20
ts = np.arange(0,t_max,dt)

states = np.zeros((len(ts), 4))
states[0] = [1, 0, 0, 1]
x = np.zeros(len(ts))
for i in range(1,len(ts)):
    states[i] = RK4_step(F, states[i-1], ts[i-1], dt)
    x[i-1] = states[i-1][0]
    
x[-1] = states[-1][0]

## 2.a   
### T simulación
transformada = np.fft.rfft(x)
fs = np.fft.rfftfreq(len(x))

f = fs[np.argmax(abs(transformada))]
eq_s = 2.41888e-17
f_s = f/eq_s
T_sim = (10**18)/f_s

### T teórico
a = 5.29177e-11
m_e = 9.10938e-31
e = 1.60218e-19
k = 8.98755e9
C = (4*(np.pi**2)*m_e)/(k*(e**2))
T_teo = np.sqrt(C*(a**3))*(10**18)

print(f'2.a) {T_teo = :.5f}; {T_sim = :.5f}')

## 2.b
def RK4_modificado(F,y0,t,dt):
    k1 = F(t,y0)
    k2 = F( t+dt/2, y0 + dt*k1/2 )
    k3 = F( t+dt/2, y0 + dt*k2/2  )
    k4 = F( t+dt, y0 + dt*k3  )
    sol = y0 + dt/6 * (k1+2*k2+2*k3+k4)
    alpha = 7.29735e-3
    unit = sol[2:]/np.linalg.norm(sol[2:])
    raiz = np.sqrt((np.linalg.norm(y0[2:])**2)-((4/3)*(alpha**3)*(np.linalg.norm(k1[2:])**2)*(dt)))
    sol[2:] = unit*raiz
    return sol

states_m = np.zeros((len(ts), 4))
states_m[0] = [1, 0, 0, 1]
x_m = np.zeros(len(ts))
y_m = np.zeros(len(ts))
r_m = np.zeros(len(ts))
v_m = np.zeros(len(ts))
x_m[0] = 1
y_m[0] = 0
r_m[0] = 1
v_m[0] = 1
for i in range(1, len(ts)):
    states_m[i] = RK4_modificado(F, states_m[i-1], ts[i-1], dt)
    x_m[i] = states_m[i][0]
    y_m[i] = states_m[i][1]
    r_m[i] = np.linalg.norm(states_m[i][0:2])
    v_m[i] = np.linalg.norm(states_m[i][2:])
    if np.linalg.norm(states_m[i][:2]) < 0.1:
        break

x_m = x_m[:i+1]
y_m = y_m[:i+1]
r_m = r_m[:i+1]
v_m = v_m[:i+1]
ts = ts[:i+1]

fig = plt.figure(figsize=(5,5))
plt.plot(x_m, y_m)
plt.xlim(-1.1,1.1)
plt.ylim(-1.1,1.1)
plt.savefig('2.b.XY.pdf')

def U_electrica(r):
    return (1)/r
def cinetica(v):
    return ((0.5)*(v**2))

E_t = U_electrica(r_m) + cinetica(v_m)
E_k = cinetica(v_m)

fig, (ax1, ax2, ax3) = plt.subplots(ncols= 1, nrows= 3)
ax1.plot(ts, r_m, c='r')
ax1.set_title('r vs t')
ax2.plot(ts, E_k, c='b')
ax2.set_title('K vs t')
ax3.plot(ts, E_t, c='g')
ax3.set_title('E vs t')
plt.savefig('2.b.diagnostics.pdf')
t_fall = (ts[-1]*eq_s)*(10**18)
print(f'2.b) {t_fall = :.5f}')


"""
a, e, μ, α = 0.38709893, 0.20563069, 39.4234021, 1.09778201e-8
#α = 1e-2

def mercurio_precesion(t, R, μ, α):
  x, v_x, y, v_y = R
  r = np.sqrt(x**2 + y**2)
  return np.array([v_x, - (μ*x / (r**3)) * (1 + α/(r**2)), v_y, - (μ*y / (r**3)) * (1 + α/(r**2))])

x_0, y_0, vx_0, vy_0 = a*(1+e), 0, 0, np.sqrt(( μ/a) * (1-e) / (1+e))

fig, ax = plt.subplots(figsize=(10, 10))
ts = np.arange(0, 10, 1e-4)
sol = solve_ivp(
    fun = mercurio_precesion,
    t_span=(0, 10),
    y0 = [x_0, vx_0, y_0, vy_0],
    args= [μ, 1e-2],
    t_eval = ts,
    method="Radau")

#plt.plot(sol.y[0], sol.y[2], ".-")

fig5 = plt.figure(figsize=(5,5))
plt.xlim(-1,1)
plt.ylim(-1,1)
linea, = plt.plot(sol.y[0,:1], sol.y[2,:1])
punto = plt.scatter([sol.y[0,0]],[sol.y[2,0]],s=70,c='sienna',zorder=100)
sun = plt.scatter([0], [0], s = 300, c = "yellow", zorder=100)

def frame(i):
    linea.set_data(sol.y[0,:i+1],sol.y[2,:i+1])
    punto.set_offsets([sol.y[0,i],sol.y[2,i]])
    sun.set_offsets([0, 0])
    return linea, punto, sun

animm = animation.FuncAnimation(fig5,frame,frames=range(1,len(sol.t)-1))
animm.save("3.a.mp4")

sol = solve_ivp(
    fun = mercurio_precesion,
    t_span=(0, 10),
    y0 = [x_0, vx_0, y_0, vy_0],
    args= [μ,α],
    method="Radau")

rs = []
for i in range(len(sol.y[0])): rs.append(np.sqrt(sol.y[0][i]**2 + sol.y[2][i]**2))
rs = np.array(rs)
peaks, _ = find_peaks(rs, height=0)
valleys, _ = find_peaks(-1*rs + 50, height=0)

#plt.scatter(sol.y[0][peaks], sol.y[2][peaks], c="r")
#plt.scatter(sol.y[0][valleys], sol.y[2][valleys], c="g")

def regression(parametro):
    x = ts[parametro]
    y = (3600*(180/np.pi)*np.arctan2(sol.y[2][parametro],sol.y[0][parametro]))
    f = lambda x, a, b: a*x + b
    popt, pcov = curve_fit(f, x, y)
    a, b = popt
    slope = popt[0]
    slope_std_err = np.sqrt(pcov[0, 0])
    #intercept_std_err = np.sqrt(pcov[1, 1])
    x_new = np.linspace(np.min(x), np.max(x), 100)
    y_new = a*x_new + b
    return x, y, x_new, y_new, slope, slope_std_err

fig7, ax = plt.subplots(1, 2, figsize=(10, 5))

Regresion1 = regression(valleys)
ax[0].scatter(Regresion1[0], Regresion1[1], c="g")
ax[0].plot(Regresion1[2], Regresion1[3], color='orange', linewidth=1)
ax[0].set_title("Periastro = {}".format(round(Regresion1[4], 2)) + "±{}".format(round(Regresion1[5], 2)))

Regresion2 = regression(peaks)
ax[1].scatter(Regresion2[0], Regresion2[1], c="b")
ax[1].plot(Regresion2[2], Regresion2[3], color='orange', linewidth=1)
ax[1].set_title("Apoastro = {}".format(round(Regresion2[4], 2)) + "±{}".format(round(Regresion2[5], 2)))

plt.savefig("3.b.pdf")
"""


def f_prime(x,f, E):
    f1,f2 = f
    df1=f2
    df2=f1*((x**2)-(2*E))
    return np.array([df1,df2])

def evento(x,f, E):
    f1,f2 = f
    return np.sqrt(f1**2+f2**2)-0.025

evento.terminal=True

def pruebas(energias_p,ci):
    energias=set()
    for E in energias_p:
      if len(energias)>4:
        break
      sol = solve_ivp(f_prime,(0,6),ci,max_step=1e-2,method="RK45",events=[evento], args=(E,))
      if sol.status==1:
        energias.add(round(E,1))
    return energias

energias_p=np.arange(0.1,15,0.01)
energias_s=pruebas(energias_p, [1,0])
energias_a=pruebas(energias_p, [0,1])

colors=["#4040BC","#1070E9","#69E6A2","#FBF988", "#ED4233", "#4151EB","#57CEE0", "#9AE857", "#EBAF57", "#B72925"]
fig, ax = plt.subplots()
x=np.linspace(-6,6,602)
ax.plot(x*(-1),(3/4*x)**2, linestyle="--", color="#CECECE")
for i in range(len(energias_s)):
  E=list(energias_s)[i]
  sol = solve_ivp(f_prime,(0,6),[1,0],max_step=1e-2,method="RK45", args=(E,))
  ax.axhline(E, color="#EDEDED")
  if i==0 or i==2:
    ax.plot(sol.t*(-1),sol.y[0]*1/4+E,color=colors[i])
    ax.plot(sol.t,sol.y[0]*1/4+E, color=colors[i])
  else:
    ax.plot(sol.t*(-1),sol.y[0]*(-1/4)+E,color=colors[i])
    ax.plot(sol.t,sol.y[0]*(-1/4)+E, color=colors[i])
for i in range(len(energias_a)):
  E=list(energias_a)[i]
  sol = solve_ivp(f_prime,(0,6),[0,1],max_step=1e-2,method="RK45", args=(E,))
  ax.axhline(E, color="#EDEDED")
  if i==2:
    ax.plot(sol.t*(-1),sol.y[0]*(3/4)+E, color=colors[i+5])
    ax.plot(sol.t,sol.y[0]*(-3/4)+E, color=colors[i+5])
  else:
    ax.plot(sol.t*(-1),sol.y[0]*(-3/4)+E, color=colors[i+5])
    ax.plot(sol.t,sol.y[0]*(3/4)+E, color=colors[i+5])
plt.ylim(0,11)
plt.xlim(-6,6)
plt.ylabel("Energía")
plt.savefig("4.pdf")