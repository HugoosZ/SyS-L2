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

k = 2  # Factor de proporcionalidad aplicado
Desplazamiento = 1  # Corrimiento temporal (en segundos)

# Onda triangular
triangular = Am * signal.sawtooth(2 * np.pi * f * t , 0.5)

# Aplicar corrimiento temporal a las respuestas a impulso
t_aplicado = t - Desplazamiento  # Aplicar corrimiento temporal hacia la derecha

# Exponencial decreciente con factor de proporcionalidad y corrimiento
exp_decreciente = k * np.exp(-a * t_aplicado) * ((np.heaviside(t_aplicado, 1)) - (np.heaviside(t_aplicado - 1, 1)))

#impulso = k * impulsoArr(t,Desplazamiento)

cero = np.argmin(np.abs(t - Desplazamiento))
impulso = np.zeros_like(t)
impulso[cero] = 1

# Convoluciones
conv_impulso_triangular = np.convolve(triangular, impulso, mode='same')

# Graficar las señales y sus convoluciones
plt.figure(figsize=(16, 10))

# Convolucion señal Sierra con Escalón

plt.subplot(3, 1, 1)
plt.plot(t, triangular)
plt.title(f'Convolución Sierra y Escalón (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

# Convolucion señal Impulso con Triangular
plt.subplot(3, 1, 2)
plt.plot(t, impulso)
plt.title(f'Convolución Impulso y Triangular (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(t, conv_impulso_triangular)
plt.title(f'Convolución Impulso y Triangular (k={k}, Corrimiento={Desplazamiento}s)')
plt.grid(True)

plt.tight_layout()
plt.show()