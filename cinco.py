import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -2  # Tiempo inicial
Tf = 2  # Tiempo final
Nm = 1000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)
a = 1  # Factor de proporcionalidad de función exponencial

# Onda senoidal
senoidal = Am * np.sin(2 * np.pi * f * t)

# Onda cuadrada
cuadrada = Am * signal.square(2 * np.pi * f * t)

# Onda triangular
triangular = Am * signal.sawtooth(2 * np.pi * f * t, 0.5)

# Onda de diente de sierra
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)

# Exponencial decreciente
exp_decreciente = np.exp(-a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))

# Exponencial creciente
exp_creciente = np.exp(a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))

# Impulso
cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[cero] = 1

# Escalón
escalon = np.heaviside(t, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_triangular_impulso = np.convolve(triangular, impulso, mode='same')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# Nuevas combinaciones
conv_seno_impulso = np.convolve(senoidal, impulso, mode='same')
conv_triangular_escalon = np.convolve(cuadrada, escalon, mode='full')[:len(t)]
conv_triangular_exp_creciente = np.convolve(triangular, exp_creciente, mode='full')[:len(t)]
conv_sierra_exp_decreciente = np.convolve(diente_sierra, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_impulso = np.convolve(cuadrada, impulso, mode='same')

# Grafico de convoluciones
plt.figure(figsize=(16, 10)) 

plt.subplot(4, 3, 1)
plt.plot(t, conv_seno_expon_decreciente)
plt.title('Senoidal con Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 3, 2)
plt.plot(t, conv_cuadrada_expon_creciente)
plt.title('Cuadrada con Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 3, 3)
plt.plot(t, conv_triangular_impulso)
plt.title('Triangular con Impulso')
plt.grid(True)

plt.subplot(4, 3, 4)
plt.plot(t, conv_sierra_escalon)
plt.title('Sierra con Escalón')
plt.grid(True)

plt.subplot(4, 3, 5)
plt.plot(t, conv_seno_impulso)
plt.title('Senoidal con Impulso')
plt.grid(True)

plt.subplot(4, 3, 6)
plt.plot(t, conv_triangular_escalon)
plt.title('Triangular con Escalón')
plt.grid(True)

plt.subplot(4, 3, 7)
plt.plot(t, conv_triangular_exp_creciente)
plt.title('Triangular con Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 3, 8)
plt.plot(t, conv_sierra_exp_decreciente)
plt.title('Sierra con Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 3, 9)
plt.plot(t, conv_cuadrada_impulso)
plt.title('Cuadrada con Impulso')
plt.grid(True)

plt.tight_layout()
plt.show()