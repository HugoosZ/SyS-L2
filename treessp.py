import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
A = 1  # Amplitud de la onda
Ti = -1  # Tiempo inicial
Tf = 1  # Tiempo final
Nm = 1000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)

# Onda senoidal
senoidal = A * np.sin(2 * np.pi * f * t)

# Onda cuadrada
cuadrada = A * signal.square(2 * np.pi * f * t)

# Onda triangular
triangular = A * signal.sawtooth(2 * np.pi * f * t, 0.5)

# Onda de diente de sierra
diente_sierra = A * signal.sawtooth(2 * np.pi * f * t)



# Exponencial decreciente
exponencial_decreciente = np.exp(-t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))


# Exponencial creciente
exponencial_creciente = np.exp(t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))


# Impulso
timpulso = np.linspace(0, Ti , Nm)
def ddf(t,sig):
    val = np.zeros_like(t)
    val[(-(1/(2*sig))<=t) & (t<=(1/(2*sig)))] = 1
    return val

sig=1000

impulso = ddf(timpulso,sig)

# Escalón
escalon_t = np.heaviside(t, 1)


# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exponencial_decreciente, mode='full')
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exponencial_creciente, mode='full')
conv_impulso_triangular = np.convolve(impulso, triangular, mode='full')
conv_sierra_escalon = np.convolve(diente_sierra, escalon_t, mode='full')

# Graficar las señales y sus convoluciones
plt.figure(figsize=(12, 12))

# Senoidal con Exponencial Decreciente
plt.subplot(4, 3, 1)
plt.plot(t, senoidal, label='Onda Senoidal')
plt.title('Onda Senoidal')
plt.grid(True)

plt.subplot(4, 3, 2)
plt.plot(t, exponencial_decreciente, label='Exponencial Decreciente')
plt.title('Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 3, 3)
plt.plot(t, conv_seno_expon_decreciente[:Nm])
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

# Cuadrada con Exponencial Creciente
plt.subplot(4, 3, 4)
plt.plot(t, cuadrada, label='Onda Cuadrada')
plt.title('Onda Cuadrada')
plt.grid(True)

plt.subplot(4, 3, 5)
plt.plot(t, exponencial_creciente, label='Exponencial Creciente')
plt.title('Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 3, 6)
plt.plot(t, conv_cuadrada_expon_creciente[:Nm])
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

# Impulso con Triangular
plt.subplot(4, 3, 7)
plt.plot(t, triangular, label='Onda Triangular')
plt.title('Onda Triangular')
plt.grid(True)

plt.subplot(4, 3, 8)
plt.plot(t, impulso, label='Impulso')
plt.title('Impulso Unitario')
plt.grid(True)


plt.subplot(4, 3, 9)
plt.plot(t, conv_impulso_triangular[:Nm])
plt.title('Convolución Impulso y Triangular')
plt.grid(True)

# Sierra con Escalón
plt.subplot(4, 3, 10)
plt.plot(t, diente_sierra, label='Onda de Sierra')
plt.title('Onda de Diente de Sierra')
plt.grid(True)

plt.subplot(4, 3, 11)
plt.plot(t, escalon_t, label='Escalón')
plt.title('Escalón')
plt.grid(True)

plt.subplot(4, 3, 12)
plt.plot(t, conv_sierra_escalon[:Nm])
plt.title('Convolución Sierra y Escalón')
plt.grid(True)

plt.tight_layout()
plt.show()
