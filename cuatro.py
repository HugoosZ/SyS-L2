import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros para las señales
fs = 1000  # Frecuencia de muestreo
T = 1  # Duración en segundos
t_periodica = np.linspace(0, T, int(fs * T), endpoint=False)
t_aperiodica = np.linspace(-1, 2, int(fs * 3), endpoint=False)

# Factores de proporcionalidad y corrimiento temporal
k = 2  # Factor de proporcionalidad 
corrimiento = 0.1  # Corrimiento temporal 

# Ajustar el vector de tiempo para aplicar el corrimiento
t_aperiodica_corrida = t_aperiodica - corrimiento  # Corrimiento hacia la derecha

# Señales Periódicas
senoidal = np.sin(2 * np.pi * 5 * t_periodica)  # Señal senoidal
cuadrada = signal.square(2 * np.pi * 5 * t_periodica)  # Señal cuadrada
triangular = signal.sawtooth(2 * np.pi * 5 * t_periodica, 0.5)  # Señal triangular
diente_sierra = signal.sawtooth(2 * np.pi * 5 * t_periodica)  # Señal diente de sierra

# Señales Aperiódicas modificadas con el factor de proporcionalidad y corrimiento temporal
exp_decreciente = k * np.exp(-t_aperiodica_corrida) * ((t_aperiodica_corrida >= 0) & (t_aperiodica_corrida < 1))
exp_creciente = k * np.exp(t_aperiodica_corrida) * ((t_aperiodica_corrida >= 0) & (t_aperiodica_corrida < 1))

# Mejorar la aproximación de la función impulso
impulso = k * np.where(np.abs(t_aperiodica_corrida) < 0.01, 1, 0)  # Impulso como una ventana más pequeña

escalon = k * np.where(t_aperiodica_corrida >= 0, 1, 0)

# Realizar convoluciones (cada señal periódica con una aperiódica modificada)
conv_senoidal_exp_decreciente = np.convolve(senoidal, exp_decreciente, mode='same')[:len(t_periodica)]
conv_cuadrada_exp_creciente = np.convolve(cuadrada, exp_creciente, mode='same')[:len(t_periodica)]
conv_triangular_impulso = np.convolve(triangular, impulso, mode='same')[:len(t_periodica)]
conv_diente_sierra_escalon = np.convolve(diente_sierra, escalon, mode='same')[:len(t_periodica)]

# Graficar las señales convolucionadas con el factor de proporcionalidad y corrimiento temporal
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t_periodica, conv_senoidal_exp_decreciente)
plt.title(f'Convolución: Señal Senoidal con Exp. Decreciente (k={k}, Corrimiento={corrimiento}s)')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t_periodica, conv_cuadrada_exp_creciente)
plt.title(f'Convolución: Señal Cuadrada con Exp. Creciente (k={k}, Corrimiento={corrimiento}s)')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t_periodica, conv_triangular_impulso)
plt.title(f'Convolución: Señal Triangular con Impulso (k={k}, Corrimiento={corrimiento}s)')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t_periodica, conv_diente_sierra_escalon)
plt.title(f'Convolución: Señal Diente de Sierra con Escalón (k={k}, Corrimiento={corrimiento}s)')
plt.grid(True)

plt.tight_layout()
plt.show()