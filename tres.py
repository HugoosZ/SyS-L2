import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
frecuencia = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Nm = 3000  # Número de muestras
t = np.linspace(-1, 2, Nm)

# Onda senoidal
senoidal = Am * np.sin(2 * np.pi * frecuencia * t)

# Onda cuadrada
cuadrada = Am * signal.square(2 * np.pi * frecuencia * t)

# Onda triangular
triangular = Am * signal.sawtooth(2 * np.pi * frecuencia * t, 0.5)

# Onda de diente de sierra
diente_sierra = Am * signal.sawtooth(2 * np.pi * frecuencia * t)



# Exponencial decreciente
exp_decreciente = np.exp(-t) * (np.heaviside(t, 1) - np.heaviside(t - 1, 1))

# Exponencial creciente
exp_creciente = np.exp(t) * (np.heaviside(t, 1) - np.heaviside(t - 1, 1))

# Impulso
index_cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[index_cero] = 1

# Escalón
escalon = np.heaviside(t,1)

# Convoluciones
conv_sin_expdecreciente = np.convolve(senoidal, exp_decreciente, mode='full')
conv_cuadrada_expcreciente = np.convolve(cuadrada, exp_creciente, mode='full')
conv_triangular_impulso = np.convolve(triangular, impulso, mode='same')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')

# Graficar las señales y sus convoluciones
plt.figure(figsize=(16, 10))

# Senoidal con Exponencial Decreciente
plt.subplot(4, 3, 1)
plt.plot(t, senoidal)
plt.title('Onda Senoidal')
plt.grid(True)

plt.subplot(4, 3, 2)
plt.plot(t, exp_decreciente)
plt.title('Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 3, 3)
plt.plot(t, conv_sin_expdecreciente[:Nm])
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

# Cuadrada con Exponencial Creciente
plt.subplot(4, 3, 4)
plt.plot(t, cuadrada, label='Onda Cuadrada')
plt.title('Onda Cuadrada')
plt.grid(True)

plt.subplot(4, 3, 5)
plt.plot(t, exp_creciente, label='Exponencial Creciente')
plt.title('Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 3, 6)
plt.plot(t, conv_cuadrada_expcreciente[:Nm])
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

# Impulso con Triangular
plt.subplot(4, 3, 7)
plt.plot(t, impulso, label='Impulso')
plt.title('Impulso Unitario')
plt.grid(True)

plt.subplot(4, 3, 8)
plt.plot(t, triangular, label='Onda Triangular')
plt.title('Onda Triangular')
plt.grid(True)

plt.subplot(4, 3, 9)
plt.plot(t, conv_triangular_impulso)
plt.title('Convolución Impulso y Triangular')
plt.grid(True)

# Sierra con Escalón
plt.subplot(4, 3, 10)
plt.plot(t, diente_sierra, label='Onda de Sierra')
plt.title('Onda de Diente de Sierra')
plt.grid(True)

plt.subplot(4, 3, 11)
plt.plot(t, escalon, label='Escalón')
plt.title('Escalón')
plt.grid(True)

plt.subplot(4, 3, 12)
plt.plot(t, conv_sierra_escalon[:Nm])
plt.title('Convolución Sierra y Escalón')
plt.grid(True)

plt.tight_layout()
plt.show()
