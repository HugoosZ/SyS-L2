import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -1  # Tiempo inicial
Tf = 1  # Tiempo final
Nm = 1000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)
a = 1  # Factor de proporcionalidad de función exponencial

# Onda senoidal
senoidal = Am * np.sin(2 * np.pi * f * t)

# Onda cuadrada
cuadrada = Am * signal.square(2 * np.pi * f * t)

# Exponencial decreciente
exp_decreciente = np.exp(-a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))

# Impulso
index_cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[index_cero] = 1

# Linealidad
a1, a2 = 2, 3  # Factores de proporcionalidad
combinacion = a1 * senoidal + a2 * cuadrada

# Convolución de la combinación lineal de las señales con la respuesta exponencial decreciente
conv_combinacion = np.convolve(combinacion, exp_decreciente, mode='full')[:len(t)]

# Convoluciones individuales de cada señal
conv_seno = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada = np.convolve(cuadrada, exp_decreciente, mode='full')[:len(t)]

# Combinación lineal de las convoluciones individuales
conv_linealidad = a1 * conv_seno + a2 * conv_cuadrada

# Invariancia 
Desplazamiento = 1 

senoidal_shift = Am * np.sin((2 * np.pi * f * t) -Desplazamiento)  # Señal desplazada en el tiempo

# Convolución de la señal desplazada 
conv_shift = np.convolve(senoidal_shift, impulso, mode='same')[:len(t)]


# Graficos
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.plot(t, conv_combinacion)
plt.title('Multiplicacion de Constante Pre Convolucion')

plt.grid(True)
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(t, conv_linealidad)
plt.title('Multiplicacion de Constante Post Convolucion')

plt.grid(True)
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(t, senoidal_shift)
plt.title('Invariancia Temporal Pre Convolucion')

plt.grid(True)
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(t, conv_shift)
plt.title('Invariancia Temporal Post Convolucion')
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()

