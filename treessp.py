import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
frecuencia = 2  # Frecuencia de la onda
amplitud = 1  # Amplitud de la onda
duracion = 1  # Duración en segundos
muestras = 10000  # Número de muestras
t = np.linspace(-1, duracion+1, muestras,False)

# Onda senoidal
senoidal = amplitud * np.sin(2 * np.pi * frecuencia * t)

# Onda cuadrada
cuadrada = amplitud * signal.square(2 * np.pi * frecuencia * t)

# Onda triangular
triangular = amplitud * signal.sawtooth(2 * np.pi * frecuencia * t, 0.5)

# Onda de diente de sierra
diente_sierra = amplitud * signal.sawtooth(2 * np.pi * frecuencia * t)

# Exponencial decreciente: e^(-t)[u(t) - u(t - 1)]
exp_decreciente = np.exp(-t) * (np.heaviside(t, 1) - np.heaviside(t - 1, 1))

# Exponencial creciente: e^(t)[u(t) - u(t - 1)]
exponencial_creciente = np.exp(t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))

# Impulso: δ(t)
timpulso = np.linspace(-1, duracion, muestras,False)


def ddf(t,sig):
    val = np.zeros_like(t)
    val[(-(1/(2*sig))<=t) & (t<=(1/(2*sig)))] = 1
    return val

sig=1000
impulso = ddf(t,sig)
# Escalón: u(t)
escalon = np.heaviside(t, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exponencial_creciente, mode='full')
conv_impulso_triangular = np.convolve(impulso, triangular, mode='full')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')

# Graficar las señales y sus convoluciones
plt.figure(figsize=(12, 12))
T = np.linspace(-1, duracion+1, muestras,False)


# Senoidal con Exponencial Decreciente
plt.subplot(4, 3, 1)
plt.plot(T, senoidal, label='Onda Senoidal')
plt.title('Onda Senoidal')
plt.grid(True)

plt.subplot(4, 3, 2)
plt.plot(T, exp_decreciente, label='Exponencial Decreciente')
plt.title('Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 3, 3)
plt.plot(T, conv_seno_expon_decreciente[:muestras])
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

# Cuadrada con Exponencial Creciente
plt.subplot(4, 3, 4)
plt.plot(T, cuadrada, label='Onda Cuadrada')
plt.title('Onda Cuadrada')
plt.grid(True)

plt.subplot(4, 3, 5)
plt.plot(T, exponencial_creciente, label='Exponencial Creciente')
plt.title('Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 3, 6)
plt.plot(T, conv_cuadrada_expon_creciente[:muestras])
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

# Impulso con Triangular
plt.subplot(4, 3, 7)
plt.plot(T, impulso, label='Impulso')
plt.title('Impulso Unitario')
plt.grid(True)

plt.subplot(4, 3, 8)
plt.plot(T, triangular, label='Onda Triangular')
plt.title('Onda Triangular')
plt.grid(True)

plt.subplot(4, 3, 9)
plt.plot(T, conv_impulso_triangular[:muestras])
plt.title('Convolución Impulso y Triangular')
plt.grid(True)

# Sierra con Escalón
plt.subplot(4, 3, 10)
plt.plot(T, diente_sierra, label='Onda de Sierra')
plt.title('Onda de Diente de Sierra')
plt.grid(True)

plt.subplot(4, 3, 11)
plt.plot(T, escalon, label='Escalón')
plt.title('Escalón')
plt.grid(True)

plt.subplot(4, 3, 12)
plt.plot(T, conv_sierra_escalon[:muestras])
plt.title('Convolución Sierra y Escalón')
plt.grid(True)

plt.subplot(4, 3, 9)
plt.plot(T, conv_impulso_triangular[:muestras])
plt.title('Convolución Impulso y Triangular')
plt.grid(True)


plt.tight_layout()
plt.show()
