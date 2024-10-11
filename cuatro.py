import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -1  # Tiempo inicial
Tf = 2  # Tiempo final
Nm = 3000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)
a = 1  # Factor de proporcionalidad de función exponencial
k = 2  # Factor de proporcionalidad aplicado
Desplazamiento = 1  # Corrimiento temporal (en segundos)

# Onda senoidal
senoidal = Am * np.sin(2 * np.pi * f * t)

# Onda cuadrada
cuadrada = Am * signal.square(2 * np.pi * f * t)

# Onda triangular
triangular = Am * signal.sawtooth(2 * np.pi * f * t + Desplazamiento, 0.5)

# Onda de diente de sierra
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)

# Aplicar corrimiento temporal a las respuestas a impulso
t_aplicado = t - Desplazamiento  # Aplicar corrimiento temporal hacia la derecha

# Exponencial decreciente con factor de proporcionalidad y corrimiento
exp_decreciente = k * np.exp(-a * t_aplicado) * ((np.heaviside(t_aplicado, 1)) - (np.heaviside(t_aplicado - 1, 1)))

# Exponencial creciente con factor de proporcionalidad y corrimiento
exp_creciente = k * np.exp(a * t_aplicado) * ((np.heaviside(t_aplicado, 1)) - (np.heaviside(t_aplicado - 1, 1)))


#impulso = k * impulsoArr(t,Desplazamiento)
impulso = k * np.where(t_aplicado == np.min(t_aplicado), 1, 0)

# Escalón con factor de proporcionalidad
escalon = k * np.heaviside(t_aplicado, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_impulso_triangular = np.convolve(triangular, impulso, mode='full')[:len(t)]
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# Graficar las señales y sus convoluciones
plt.figure(figsize=(16, 10))

# Convolucion señal Senoidal con Exponencial Decreciente
plt.subplot(4, 1, 1)
plt.plot(t, conv_seno_expon_decreciente)
plt.title(f'Convolución Senoidal y Exponencial Decreciente (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

# Convolucion señal Cuadrada con Exponencial Creciente
plt.subplot(4, 1, 2)
plt.plot(t, conv_cuadrada_expon_creciente)
plt.title(f'Convolución Cuadrada y Exponencial Creciente (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

# Convolucion señal Impulso con Triangular
plt.subplot(4, 1, 3)
plt.plot(t, conv_impulso_triangular)
plt.title(f'Convolución Impulso y Triangular (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

# Convolucion señal Sierra con Escalón
plt.subplot(4, 1, 4)
plt.plot(t, conv_sierra_escalon)
plt.title(f'Convolución Sierra y Escalón (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.tight_layout()
plt.show()