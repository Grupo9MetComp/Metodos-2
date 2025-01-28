import numpy as np
import matplotlib.pyplot as plt
import csv
import scipy as sp
datos = np.genfromtxt('Rhodium.csv', delimiter=',')

datos = np.delete(datos, 0, 0)

datos_eliminados = 0
datos_limpieza = datos.copy()
for i in range(1, 1198):
    if datos_limpieza[i + 1, 1] > 1.1 * datos_limpieza[i, 1]:
        datos_limpieza[i + 1, 1] = (datos_limpieza[i, 1] + datos_limpieza[i + 2, 1]) / 2
        datos_eliminados += 1

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21/1.5, 9/1.5))

ax1.plot(datos[:, 0], datos[:, 1], '.-', label="Datos corruptos")
ax1.set_title("Datos corruptos")
ax1.set_xlabel("Longitud de onda")
ax1.set_ylabel("Intensidad")
ax1.legend()

ax2.plot(datos_limpieza[:, 0], datos_limpieza[:, 1], label="Datos limpios",color='Green')
ax2.set_title("Datos limpios")
ax2.set_xlabel("Longitud de onda")
ax2.set_ylabel("Intensidad")
ax2.legend()

plt.tight_layout()
plt.savefig("limpieza.pdf", format="pdf")

print(f'1.a) Número de datos eliminados: {datos_eliminados}')


def fit_data(x):
    slope = (datos[415,1] - datos[260,1])/ (datos[415,0] - datos[260,0])
    intersec = datos[415,1] - slope*datos[415,0]
    return slope*x + intersec
    
datos_espectro = datos.copy()

for i in range(260,416):
    datos_espectro[i,1] = fit_data(datos_espectro[i,0])
    
plt.figure(figsize=(9, 6), dpi=150)
plt.plot(datos_espectro[:, 0], datos_espectro[:, 1], label="Datos aislados",color='purple')
plt.title("Datos espectro fondo")
plt.xlabel("Longitud de onda")
plt.ylabel("Intensidad")
plt.legend()
plt.grid(True)


def calculate_fwhm(data):
    max_y = data[:, 1].max()
    half_max = max_y / 2

    indices = np.where(data[:, 1] >= half_max)[0]

    x_left = data[indices[0], 0]
    x_right = data[indices[-1], 0]

    return round(x_right - x_left,4) , round(max_y,4)


peak_espectro = calculate_fwhm(datos_espectro)


xs = datos[:, 0]
ys = datos[:, 1] - datos_espectro[:, 1]
ys = np.where(ys < 0, 0, ys)

plt.figure(figsize=(9, 6), dpi=150)
plt.plot(xs,ys,label='Picos aislados',color='violet')
plt.title("Picos aislados")
plt.xlabel("Longitud de onda")
plt.ylabel("Intensidad")
plt.legend()
plt.grid(True)
plt.savefig("picos.pdf", format="pdf")



print(f'1.b) Metodo: Recortar, linea, restar, separar')

def calculate_fwhm2(xs, ys):
    max_y = ys.max()
    half_max = max_y / 2

    indices = np.where(ys >= half_max)[0]

    x_left = xs[indices[0]]
    x_right = xs[indices[-1]]

    return round(x_right - x_left,4) , round(max_y,4)


peak1_mask = (xs >= 80) & (xs <= 100)
peak2_mask = (xs >= 105) & (xs <= 125)

xs_peak1, ys_peak1 = xs[peak1_mask], ys[peak1_mask]
xs_peak2, ys_peak2 = xs[peak2_mask], ys[peak2_mask]


fwhm_peak1 = calculate_fwhm2(xs_peak1, ys_peak1)
fwhm_peak2 = calculate_fwhm2(xs_peak2, ys_peak2)

print(f'1.c) Para el espectro de fondo, su FWHF es {peak_espectro[0]} y su maximo respectovo es {peak_espectro[1]}. Para el primer pico su FWHM y Maximo correspondientes son: {fwhm_peak1[0]} y {fwhm_peak1[1]}. Para el pico dos su FWHM y maximo correspondientes son: {fwhm_peak2[0]} y {fwhm_peak2[1]}')

t = []
B = []
H = []

with open('hysteresis.dat', newline='') as file:
    reader = csv.reader(file, delimiter=' ')
    for row in reader:
        if len(row) != 3:
            if len(row) == 1:
                datos = row[0].split('-')
                if len(datos) == 2:
                    b = datos[1][:5]
                    h = datos[1][5:]
                    datos.pop(1)
                    datos.insert(1, b)
                    datos.insert(2, h)
                t.append(float(datos[0]))
                B.append(-float(datos[1]))
                H.append(-float(datos[2]))
            elif len(row) == 2:
                if (len(row[1]) > 5) and ('-' not in row[1]):
                    b = row[1][:5]
                    h = row[1][5:]
                    row.pop()
                    row.append(b)
                    row.append(h)
                else:
                    for i in range(2):
                        if '-' in row[i]:
                            datos = row[i].split('-')
                            row.pop(i)
                            row.insert(i, datos[0])
                            row.insert(i+1, ('-'+datos[1]))
                            break
                t.append(float(row[0]))
                B.append(float(row[1]))
                H.append(float(row[2]))
        else:
            t.append(float(row[0]))
            B.append(float(row[1]))
            H.append(float(row[2]))

plt.plot(t, B, label='B (mT)')
plt.plot(t, H, label='H (A/m)')
plt.title('Datos vs tiempo')
plt.xlabel('t (ms)')
plt.legend()
plt.savefig('histérico.pdf')


t = np.array(t)
dt = t[1]-t[0]
a = 1/dt
B = np.array(B)
fs = np.linspace(0, 3, 1000)
F = np.zeros_like(fs, dtype=np.complex128)
for i, f in enumerate(fs):
    F[i] = np.mean(B*np.exp(-2j*np.pi*a*t*f))
 
f = fs[np.argmax(abs(F))]

print("2.b) La frecuencia de oscilación de B es de {} Hz. Este resultado se obtuvo aplicando una transformada de Fourier a los datos, multiplicando el exponente por el inverso del intervalo de tiempo entre dos medidas y finalmente consevando la frecuencia con la que se obtuvo un valor más alto en la transformada".format(f))