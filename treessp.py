import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -1 # Tiempo inicial
Tf = 1  # Tiempo final
Nm = 1000  # Número de muestras
t = np.linspace(Ti, Tf, Nm )
a = 1 # Factor de proporcionalidad de funcion exponencial
# Onda senoidal
senoidal = Am * np.sin(2 * np.pi * f * t)

# Onda cuadrada
cuadrada = Am * signal.square(2 * np.pi * f * t)

# Onda triangular
triangular = Am * signal.sawtooth(2 * np.pi * f * t, 0.5)

# Onda de diente de sierra
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)



# Exponencial decreciente
exp_decreciente = np.exp(-a*t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))


# Exponencial creciente
exp_creciente = np.exp(a*t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))


# Impulso
timpulso = np.linspace(0, Tf , Nm)
def ddf(t,sig):
    val = np.zeros_like(t)
    val[(-(1/(2*sig))<=t) & (t<=(1/(2*sig)))] = 1
    return val
sig=100000


index_cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[index_cero] = 1

def convolucion_manual(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M ) 
    
    # Invertir h
    h_inv = h[::1] #NO SE PORQUE FUNCIONA SIN INVERTIR
    
    # Desplazar h y realizar el producto punto a punto
    for n in range(len(y)):
        suma = 0
        for k in range(M):
            if 0 <= n - k < N:
                suma += x[n - k] * h_inv[k]
        y[n] = suma
    
    return y

# Escalón
escalon = np.heaviside(t, 1)


# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_impulso_triangular = np.convolve(triangular, impulso, mode='same')[:len(t)]
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# 2(k-3(k/3 + 1/2))
# Y[n] = sum[-1,2] x[k]h[n-k] ->x[k] = diente, h[n-k] = escalon
# Y[n] = sum[0,2] diente[k]escalon[n-k] -> {n-k>=0} -> {n>=k}
# Y[0] = sum[0,2] 2(0-3(0/3 + 1/2))
# Y[1] = sum[0,2] 2(1-3(1/3 + 1/2))
# Y[2] = sum[0,2] 2(2-3(2/3 + 1/2))

# Convoluciones manuales 

conv1 = convolucion_manual(senoidal, exp_decreciente)[:len(t)]
conv2 = convolucion_manual(cuadrada, exp_creciente)[:len(t)]
conv3 = convolucion_manual(triangular, impulso)[:len(t)]
conv4 = convolucion_manual(diente_sierra, escalon)[:len(t)]



# Grafico de las señales y sus convoluciones
plt.figure(figsize=(16, 10))

# Senoidal con Exponencial Decreciente


plt.subplot(4, 2, 1)
plt.plot(t, conv_seno_expon_decreciente)
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 2, 2)
plt.plot(t, conv1)
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

# Cuadrada con Exponencial Creciente

plt.subplot(4, 2, 3)
plt.plot(t, conv_cuadrada_expon_creciente)
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 2, 4)
plt.plot(t, conv2)
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

# Impulso con Triangular


plt.subplot(4, 2, 5)
plt.plot(t, conv_impulso_triangular)
plt.title('Convolución Impulso y Triangular')
plt.grid(True)

plt.subplot(4, 2, 6)
plt.plot(t, conv3)
plt.title('Convolución Impulso y Triangular')
plt.grid(True)

# Sierra con Escalón


plt.subplot(4, 2, 7)
plt.plot(t, conv_sierra_escalon)
plt.title('Convolución Sierra y Escalón')
plt.grid(True)


plt.subplot(4, 2, 8)
plt.plot(t, conv4)
plt.title('Convolución Sierra y Escalón')
plt.grid(True)

plt.tight_layout()
plt.show()
