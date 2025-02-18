import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from PIL import Image
from numpy import fft
import scipy as sp
from numpy.typing import NDArray

# 1 a y b
def datos_prueba(t_max:float, dt:float, amplitudes:NDArray[float], frecuencias:NDArray[float], ruido:float=0.0) -> NDArray[float]:
    ts = np.arange(0.,t_max,dt)
    ys = np.zeros_like(ts,dtype=float)
    for A,f in zip(amplitudes,frecuencias):
        ys += A*np.sin(2*np.pi*f*ts)
        ys += np.random.normal(loc=0,size=len(ys),scale=ruido) if ruido else 0
    return ts,ys

time_vals_1, y_vals_1 = datos_prueba(20,0.01,np.arange(1,4),np.arange(1,4,1),0)
time_vals_2, y_vals_2 = datos_prueba(20,0.01,np.arange(1,4),np.arange(1,4,1),5)

fig, ax = plt.subplots(2, 1, figsize=(15, 10))


ax[0].plot(time_vals_1, y_vals_1,'.-')
ax[0].set(title='Sin ruido',xlabel = 't', ylabel = 'y')

ax[1].plot(time_vals_2, y_vals_2,'.-')
ax[1].set(title='Con ruido',xlabel = 't', ylabel = 'y')

def Fourier(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    t = np.asarray(t)
    y = np.asarray(y)
    f = np.asarray(f)
    exponentes = np.exp(-2j * np.pi * np.outer(t, f))
    transformada = np.dot(y, exponentes)

    return transformada
test_freqs = np.arange(0,10,0.05)
trans_clean = Fourier(time_vals_1, y_vals_1,test_freqs) 
trans_noise = Fourier(time_vals_2, y_vals_2,test_freqs) 

fig, ax = plt.subplots(2, 1, figsize=(10, 10))

ax[0].plot(test_freqs, np.abs(trans_clean), '.-')
ax[0].set(title='Fourier para senal sin ruido', xlabel = 'Frecuencia (Hz)' , ylabel = r" $\left| \mathcal{F}\{t_i , y_i\}(f) \right|$")
ax[1].plot(test_freqs, np.abs(trans_noise), '.-')
ax[1].set(title='Fourier para senal con ruido', xlabel = 'Frecuencia (Hz)' , ylabel = r" $\left| \mathcal{F}\{t_i , y_i\}(f) \right|$")
plt.savefig('1 a) .pdf')
print(f'1.a) La base de la transformada se vuelve muy ruidosa y aparecen otros picos falsos. Los picos reales se pueden perder en los del ruido')

#prueba para el seno a 10hercios
t = np.linspace(0,2*np.pi,400)
yv = np.sin(2*np.pi*10*t)
#plt.plot(t,yv)

f = np.arange(8,12,0.01)
trans_sin = Fourier(t,yv,f)

plt.plot(f, np.abs(trans_sin))
plt.scatter(f[200], np.abs(trans_sin)[200], color='g', label="Pico en índice 200")
plt.axvline(x=10, color='r', linestyle="--", label="Frecuencia esperada (10 Hz)")
plt.legend()
plt.xlabel("Frecuencia (Hz)")
plt.ylabel("|Transformada|")
plt.title("Espectro de Fourier")
x = np.abs(trans_sin)  
w = sp.signal.peak_widths(x, np.array([200]))

#print("FWHM:", w[0])

def generar_senal(t_max, dt, f, A=1.0):
    t = np.arange(0, t_max, dt)
    y = A * np.sin(2 * np.pi * f * t)
    return t, y

t_max_values = np.logspace(1, 2.5, num=20)
dt = 0.01  
f_real = 10  
fwhm_values = []

for t_max in t_max_values:
    t, y = generar_senal(t_max, dt, f_real)
    f = np.linspace(f_real - 2, f_real + 2, 1000)
    trans_sin = Fourier(t, y, f)
    abs_trans = np.abs(trans_sin)
    peak_idx, _ = sp.signal.find_peaks(abs_trans, height=abs_trans.max() * 0.5)

    if len(peak_idx) > 0:
        max_peak = peak_idx[np.argmax(abs_trans[peak_idx])]  
        fwhm = sp.signal.peak_widths(abs_trans, [max_peak], rel_height=0.5)[0][0]
        fwhm_values.append(fwhm)

plt.figure(figsize=(8, 6),dpi=200)
plt.loglog(t_max_values[:len(fwhm_values)], fwhm_values, 'o-', label="FWHM vs $t_{max}$")
plt.xlabel("$t_{max}$ (s)")
plt.ylabel("FWHM (Hz)")
plt.title("Ancho a media altura (FWHM) vs. Duración de la señal (Usando signal.peak_widths)")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)

plt.savefig("1.b.pdf")

# 1.c y 2.a

t,y,sigma=np.loadtxt('OGLE-LMC-CEP-0001.dat', delimiter=' ',unpack=True)
plt.plot(t,y)
plt.xlabel("t")
plt.ylabel("y")
plt.title("y vs t")

def Fourier(t: NDArray[float], y: NDArray[float], f: NDArray[float]) -> NDArray[complex]:
    t = np.asarray(t)
    y = np.asarray(y)
    f = np.asarray(f)
    exponentes = np.exp(-2j * np.pi * np.outer(t, f))
    transformada = np.dot(y, exponentes)
    return transformada

f=np.fft.rfftfreq(len(t),t[1]-t[0])
F_true=Fourier(t,y,f)
f_true=f[np.argmax(np.abs(F_true))]
plt.plot(f,np.abs(F_true),color="b", label="F_true")
plt.axvline(x=f_true, color='g', linestyle="--", label="f_true")
plt.legend()
plt.xlabel("f (Hz)")
plt.ylabel("|F|")
plt.title("Transformada vs Frecuencia")
fN=2*f_true
print(f"1.c) f Nyquist:{fN = :.5f}")
print(f"1.c) f true:{f_true = :.5f}")

phi_true=[]
for n in t:
  phi_true.append(np.mod(f_true*n, 1))
phi_true=np.array(phi_true)
plt.scatter(phi_true, y, label="Phi_true")
plt.xlabel("Phi")
plt.ylabel("H")
plt.title("H vs Phi")
plt.savefig("1.c.pdf")

t,H=np.loadtxt('H_field.csv', delimiter=',', skiprows=1,unpack=True)
plt.plot(t,H)
plt.xlabel("t")
plt.ylabel("H")
plt.title("t vs H")

f=np.fft.rfftfreq(len(H),t[1]-t[0])
F_fast=np.fft.rfft(H)
f=np.fft.rfftfreq(len(H),t[1]-t[0])
plt.plot(f,np.abs(F_fast),color="b", label="F_fast")
f_fast=f[np.argmax(np.abs(F_fast))]
f2=np.linspace(f[0],f[-1], 1000)
F_general=Fourier(t,H,f2)
plt.scatter(f2,np.abs(F_general), color="r", label="F_general")
f_general=f2[np.argmax(np.abs(F_general))]
plt.axvline(x=f_fast, color='g', linestyle="--", label="f_fast")
plt.axvline(x=f_general, color='y', linestyle=":", label="f_general")
plt.legend()
plt.xlabel("f (Hz)")
plt.ylabel("|F|")
plt.title("Transformada vs Frecuencia")
print(f"2.a) {f_fast = :.5f}; {f_general = :.5f}")

phi_fast=[]
phi_general=[]
for n in t:
  phi_fast.append(np.mod(f_fast*n, 1))
  phi_general.append(np.mod(f_general*n, 1))
phi_fast=np.array(phi_fast)
phi_general=np.array(phi_general)
plt.scatter(phi_fast, H, label="Phi_fast")
plt.scatter(phi_general, H, label="Phi_general")
plt.legend()
plt.xlabel("Phi")
plt.ylabel("H")
plt.title("H vs Phi")
plt.savefig("2.a.pdf")

# 2.b

years = []
months = []
days = []
manchas = []

with open('list_aavso-arssn_daily.txt') as file:
    reader = csv.reader(file, delimiter=' ')
    i = 0
    for row in reader:
        if i >= 2:
            vacios = 0
            for element in row:
                if element == '':
                    vacios += 1
            for i in range(vacios):
                row.remove('')
            if int(row[0]) >= 2010:
                break
            else:
                years.append(row[0])
                months.append(row[1])
                days.append(row[2])
                manchas.append(int(row[3]))
        i += 1

df = pd.DataFrame({"year":years,
                   "month": months,
                   "day": days,
                   "SSN": manchas},
                  dtype=int)

df.index = tiempo1 = pd.to_datetime(df[["year", "month", "day"]])
df = df.drop(columns=["year", "month", "day"])

# 2.b.a

transformada = np.fft.rfft(df["SSN"])
freqs = np.fft.rfftfreq(len(manchas))

f = freqs[1:][np.argmax(abs(transformada[1:]))]

f_años = f*365
T = f_años**(-1)
print(f'2.b.a) {T}')

# 2.b.b

tiempo = pd.date_range('1/1/1945', '2/17/2025')
dias = np.array([int((tiempo[i]-tiempo[0])/np.timedelta64(1, 'D')) for i in range(len(tiempo))])

M = 50
y = np.zeros_like(dias, dtype=np.float64)
for t, i in enumerate(dias):
    y[i] = np.real((1/df.size)*np.sum([(transformada[k])*np.exp(2j*np.pi*freqs[k]*t) for k in range(M)]))

print(f'2.b.b) {int(round(y[-1]))}')

plt.figure(figsize=(12, 4))
plt.scatter(tiempo1, df, alpha=0.5, s=0.8)
plt.plot(tiempo, y, color='black')
plt.ylabel("Manchas Solares")
plt.savefig('2.b.pdf')


castle = np.array(Image.open("Noisy_Smithsonian_Castle.jpg"))

FFT = fft.fftshift(fft.fft2(castle))

cmap = plt.get_cmap("grey")
cmap.set_bad((1,0,0))

x_values = [330, 356, 381]

for i in range(0, 3):
    for j in range(0, 3):
        for value in x_values:
            FFT[value + i, 411 + j] = FFT[value, 415+j]
            FFT[value + i, 611 + j] = FFT[value, 415+j]

for i in range(0, 765):
    if abs(i-382)>3:
        for j in range(0, 3):
            FFT[0 + i, 511 + j] = FFT[i, 514]
            
#plt.xlim([400, 620])
#plt.ylim([300, 400])
#plt.imshow(abs(FFT),cmap=cmap,norm="log")

I_FFT = fft.ifftshift(FFT)

new_image = fft.ifft2(I_FFT)
fig, ax31 = plt.subplots(figsize=(10, 10))
plt.imshow(new_image.real, cmap = cmap)
plt.savefig("3.b.a.png")

gato = np.array(Image.open("catto.png"))

FFT2 = fft.fftshift(fft.fft2(gato))

x2_values = [[245, 412], [257, 409], [270, 406], 
             [282, 403], [294, 399], [305, 395],
             [317, 392], [330, 388], [342, 384],
             [355, 380], [367, 377]]

x3_values = [[390, 375], [403, 369], [415, 365],
             [426, 362], [439, 357], [450, 354],
             [463, 351], [475, 348]]

for i in range(0, 3):
    for j in range(0, 50):
      for x2value in x2_values: FFT2[x2value[0] + i, x2value[1] + j] = FFT2[x2value[0] + i, x2value[1] + j + 80]
  
for i in range(0, 3):
    for j in range(0, 50):
      for x3value in x3_values: FFT2[x3value[0] + i, x3value[1] - j] = FFT2[x3value[0] + i, x3value[1] - j - 80]

#plt.imshow(abs(FFT2),cmap=cmap,norm="log")
#plt.ylim([500, 300])
#plt.xlim([300, 400])

#plt.imshow(abs(FFT2),cmap=cmap,norm="log")
I2_FFT = fft.ifftshift(FFT2)

new_image2 = fft.ifft2(I2_FFT)
fig, ax32 = plt.subplots(figsize=(10, 10))
plt.imshow(new_image2.real, cmap = cmap)
plt.savefig("3.b.b.png")