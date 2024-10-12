import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -1 # Tiempo inicial
Tf = 1# Tiempo final
Nm = 4000 # Número de muestras
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
cero = np.argmin(np.abs(t- 0))
print(cero)
impulso = np.zeros_like(t)
print (impulso)
impulso[cero] = 1

# Escalón
escalon = np.heaviside(t, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_impulso_triangular = np.convolve(triangular, impulso, mode='same')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# Grafico de las señales y sus convoluciones
plt.figure(figsize=(16, 10))

# Senoidal con Exponencial Decreciente
plt.subplot(4, 1, 1)
plt.plot(t, conv_seno_expon_decreciente)
plt.title('Convolución Senoidal y Exponencial Decreciente')
plt.grid(True)

# Cuadrada con Exponencial Creciente
plt.subplot(4, 1, 2)
plt.plot(t, conv_cuadrada_expon_creciente)
plt.title('Convolución Cuadrada y Exponencial Creciente')
plt.grid(True)

# Impulso con Triangular
plt.subplot(4, 1, 3)
plt.plot(t, conv_impulso_triangular )
plt.title('Convolución Triangular y Impulso')
plt.grid(True)


# Sierra con Escalón
plt.subplot(4, 1, 4)
plt.plot(t, conv_sierra_escalon)
plt.title('Convolución Sierra y Escalón')
plt.grid(True)

plt.tight_layout()
plt.show()
