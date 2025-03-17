import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numba import njit
import scipy as sp
import matplotlib.animation as animation
import random

rand = np.random.default_rng()

# 1 punto

## 1.a)
def metropolis(f,x0,n,sigma):
    samples = np.zeros(n)
    samples[-1] = x0
    for i in range(n):
        sample_new = samples[i-1] + rand.normal(0,sigma)
        if rand.random() < f(sample_new)/f(samples[i-1]):
            samples[i] = sample_new
        else:
            samples[i] = samples[i-1]
    return samples

@np.vectorize
def g(x, n, a):
  I = 0
  for k in range(1, n):
    I+= (np.exp(-(x-k)**2 * k) / (k**a))
  return I

@njit
def test(x, n, a):
  I = 0
  for k in range(1, n):
    I+= (np.exp(-(x-k)**2 * k) / (k**a))
  return I

samples = metropolis(lambda x: test(x,10,4/5),1,n=500000,sigma=1)

fig, ax1a = plt.subplots(1, 1, figsize=(8,5))

ax1a.hist(samples,bins=200,density=True,align="right")
ax1a.set_title("Historigrama para 500000 muestras de la distribución");
plt.savefig("1.a.pdf")

xs = np.linspace(samples.min(),samples.max(),10000)

## 1.b)
def f(x): return np.exp(-x**2)
S = 0
N = len(samples)
for i in range (N):
  S+= f(samples[i]) / test(samples[i], 20, 4/5)

A = np.sqrt(np.pi)*N / S

x=np.linspace(-1.5, 10, 1000)
y=g(x, 20, 4/5) /  A
plt.plot(x, y)
plt.hist(samples, 200, density=True)

print("1.b)", A, np.std([4.0515497952658689, A]))


# 2 punto

D = 50e-2      
lam = 670e-9  
A = 0.4e-3     
a = 0.1e-3     
d = 0.1e-2
N = 1000000
N2 = N//2

y1_min = -d/2 - a/2
y1_max = -d/2 + a/2
y1_samples = np.random.uniform(y1_min, y1_max, N2)

y2_min = d/2 - a/2
y2_max = d/2 + a/2
y2_samples = np.random.uniform(y2_min, y2_max, N2)

y_samples = np.concatenate([y1_samples, y2_samples])

x_samples = np.random.uniform(-A/2, A/2, N)
area = A*2*a

z_vals = np.linspace(-0.4e-2, 0.4e-2, 250) 
I_vals = np.zeros(len(z_vals))

def fun_I(x, y, z, D, lam):
    factor_const = np.exp(4 * np.pi * 1j * D / lam)
    fase = np.exp((np.pi * 1j) / (lam * D) * ((x - y)**2 + (z - y)**2))
    return factor_const * fase
def monte_carlo(fun_I, x_samples, y_samples, z, D, lam, area):
    f_values = fun_I(x_samples, y_samples, z, D, lam)
    integral = area*np.mean(f_values)
    return integral
def I_clasico(z, D, lam, a, d):
    theta = np.arctan(z/D)
    fac_cos = np.cos((np.pi*d/lam) * np.sin(theta))**2
    fac_sinc = np.sinc((a/lam) * np.sin(theta))**2
    return fac_cos*fac_sinc
for i, z in enumerate(z_vals):
    integral = monte_carlo(fun_I, x_samples, y_samples, z, D, lam, area)
    I_vals[i] = np.abs(integral)**2

I_norm = I_vals / np.max(I_vals)



I_clas = I_clasico(z_vals, D, lam, a, d)
I_clas_norm = I_clas / np.max(I_clas)

fig_punto_2 = plt.figure(figsize=(4*(2.5),3*(2.5)),dpi=250)
plt.plot(z_vals,I_clas_norm,label='I clasico normalizado',c='b')
plt.plot(z_vals,I_norm,label = 'I de feynman',c='r')
plt.legend()
plt.grid(True)
plt.xlabel('Valor de z')
plt.ylabel(r'$I(z)$')
fig_punto_2.savefig('2.pdf')


# 3 Punto 

plt.rcParams['animation.embed_limit'] = 2**128
plt.rcParams["axes.labelsize"] = 15
plt.rcParams["figure.figsize"] = (14,5)
plt.rcParams["animation.html"] = "jshtml"
plt.rcParams["image.origin"] = "upper"

N=150
J=0.2
beta=10
U_original=np.zeros((N,N))
for i in range(N):
  for j in range(N):
    num=random.randint(0,1)
    if num==0:
      U_original[i,j]=-1
    else:
      U_original[i,j]=1


def calcular_energia(U):
  E=0
  for i in range(1,len(U)-1):
    for j in range(1,len(U[0])-1):
      E+=(U[i,j]*U[i+1,j])+(U[i,j]*U[i,j+1])+(U[i,j]*U[i-1,j])+U[i,j-1]
  E*=(-J)
  return E

fig,ax=plt.subplots()
simulacion=ax.imshow(U_original,cmap=plt.cm.binary,origin="lower")

def draw_frame(frame):
    global U_original
    for i in range(400):
        U_new=U_original.copy()
        x=random.randint(0,N-1)
        y=random.randint(0,N-1)
        if U_original[x,y]==-1:
            U_new[x,y]=1
        else:
            U_new[x,y]=-1
        E_old=calcular_energia(U_original[x-2:x+2,y-2:y+2])
        E_new=calcular_energia(U_new[x-2:x+2,y-2:y+2])
        dif=E_new-E_old
        if dif <= 0:
            U_original=U_new.copy()
        else:
            u=random.uniform(0,1)
            if u<=np.exp((-beta)*dif):
                U_original=U_new.copy()
    simulacion.set_array(U_original)
    return simulacion,

anim=animation.FuncAnimation(fig,draw_frame,frames=500,interval=20)
anim.save("3.mp4")


# 4 Punto

fileref = open('El retrato de Dorian Gray.txt', 'r', encoding='utf8')
s = fileref.read()
fileref.close()

s = s[3242:-18498]

## Seleccionar un conjunto pequeño
s = s[:50000]

## Limpiar los caracteres no deseados
s = s.replace("\r\n","\n").replace("\n\n","#").replace("\n","").replace("#","\n\n").replace("_", " ")
s = s.replace("“", "").replace("”", "").replace("’", "").replace("  ", " ").lower()
s = s.replace('é', 'e').replace('è', 'e').replace('ê', 'e')
s = s.replace('1', '').replace('5', '').replace('2', '').replace('8', '').replace('0', '')
s = s.replace('à', 'a').replace('æ', 'ae').replace('ô', 'o').replace('ç', 'c')

## Función que implementa el modelo
def generador_texto(n):
    rows = []
    cols = []
    # Este ciclo toma las n letras enteriores y las añade a las filas y la última letra la añade a columnas
    for i in range(n, len(s)):
        n_letras = s[i-n:i]
        if n_letras not in rows:
            rows.append(n_letras)
        next_l = s[i]
        if next_l not in cols:
            cols.append(next_l)
    ## Se crea el DataFrame
    F = pd.DataFrame(np.zeros((len(rows),len(cols)),dtype=int),
        index=rows,columns=cols)
    ## Se suma uno en la casilla ij cuando después de las n letras de la fila i viene la letra de la col j
    for i in range(n, len(s)):
        F.loc[s[i-n:i],s[i]] += 1
    ## Se normaliza el DataFrame
    P = F / F.sum(axis=1).values[:,None]

    ## Se genera el texto
    texto = str(np.random.choice(rows, n)[0])

    for i in range(n, 1500-n):
        texto += P.loc[texto[i-n:i]].idxmax()
    return texto

dict_file = open("words_alpha.txt", 'r', encoding="utf8")
palabras = dict_file.read()
dict_file.close()

palabras = palabras.split('\n')

ns = np.linspace(1, 8, 8)
porcentajes = np.zeros_like(ns)

for i in range(8):
    resultado = generador_texto(i+1)
    fichero = open("gen_text_n{}.txt".format(i+1), "w", encoding="utf8")
    fichero.write(resultado)
    fichero.close()
    p_resultado = resultado.replace("\r\n","\n").replace("\n\n","#").replace("\n","").replace("#"," ").replace('!', '').replace('?', '').split(' ')
    reales = 0
    for p in p_resultado:
        if p in palabras:
            reales += 1
    porc = (reales*100)/len(p_resultado)
    porcentajes[i] = porc


fig = plt.figure()
plt.plot(ns, porcentajes, 'o-')
plt.xlabel('n')
plt.ylabel('Porcentaje')
plt.savefig('4.pdf')