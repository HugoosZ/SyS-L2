import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -3  # Tiempo inicial
Tf = 3  # Tiempo final
Nm = 3000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)

a = 1  # Factor de proporcionalidad de función exponencial
k = 2  # Factor de proporcionalidad 

# Ondas Periodicas
senoidal = Am * np.sin(2 * np.pi * f * t)
cuadrada = Am * signal.square(2 * np.pi * f * t)
triangular = Am * signal.sawtooth(2 * np.pi * f * t , 0.5)
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)


# Corrimiento temporal 
Desplazamiento = 0.5  
t_aplicado = t - Desplazamiento  # Corrimiento temporal hacia la derecha

# Ondas A-Periodicas
k = 2  # Factor de proporcionalidad 

exp_decreciente = k * np.exp(-a * t_aplicado) * ((np.heaviside(t_aplicado, 1)) - (np.heaviside(t_aplicado - 1, 1)))
exp_creciente = k * np.exp(a * t_aplicado) * ((np.heaviside(t_aplicado, 1)) - (np.heaviside(t_aplicado - 1, 1)))

cero = np.argmin(np.abs(t_aplicado - 0))
impulso = np.zeros_like(t_aplicado)
impulso[cero] = k * 1

escalon = k * np.heaviside(t_aplicado, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_impulso_triangular = np.convolve(triangular, impulso, mode='same')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# Graficos
plt.figure(figsize=(16, 10))

plt.subplot(4, 1, 1)
plt.plot(t, conv_seno_expon_decreciente)
plt.title(f'Convolución Senoidal y Exponencial Decreciente (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, conv_cuadrada_expon_creciente)
plt.title(f'Convolución Cuadrada y Exponencial Creciente (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, conv_impulso_triangular)
plt.title(f'Convolución Impulso y Triangular (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, conv_sierra_escalon)
plt.title(f'Convolución Sierra y Escalón (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.tight_layout()
plt.show()