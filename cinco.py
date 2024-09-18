import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros para las señales
fs = 1000  # Frecuencia de muestreo
T = 1  # Duración en segundos
t_periodica = np.linspace(0, T, int(fs * T), endpoint=False)
t_aperiodica = np.linspace(-1, 2, int(fs * 3), endpoint=False)

# Factores de proporcionalidad y corrimiento temporal
k = 1  # Factor de proporcionalidad ajustado para las señales
corrimiento = 0.05  # Corrimiento temporal reducido

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
impulso = k * np.where(np.abs(t_aperiodica_corrida) < 0.01, 1, 0)  # Impulso como una ventana más pequeña
escalon = k * np.where(t_aperiodica_corrida >= 0, 1, 0)

# Combinaciones de convoluciones intercambiadas
conv_combinacion_1 = np.convolve(exp_decreciente, cuadrada, mode='same')[:len(t_periodica)]
conv_combinacion_2 = np.convolve(exp_creciente, senoidal, mode='same')[:len(t_periodica)]
conv_combinacion_3 = np.convolve(impulso, diente_sierra, mode='same')[:len(t_periodica)]
conv_combinacion_4 = np.convolve(escalon, triangular, mode='same')[:len(t_periodica)]
conv_combinacion_5 = np.convolve(exp_decreciente, diente_sierra, mode='same')[:len(t_periodica)]

# Graficar las 5 combinaciones
plt.figure(figsize=(12, 12))

plt.subplot(5, 1, 1)
plt.plot(t_periodica, conv_combinacion_1)
plt.title('Convolución: Exp. Decreciente con Cuadrada')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t_periodica, conv_combinacion_2)
plt.title('Convolución: Exp. Creciente con Senoidal')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(t_periodica, conv_combinacion_3)
plt.title('Convolución: Impulso con Diente de Sierra')
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t_periodica, conv_combinacion_4)
plt.title('Convolución: Escalón con Triangular')
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(t_periodica, conv_combinacion_5)
plt.title('Convolución: Exp. Decreciente con Diente de Sierra')
plt.grid(True)

plt.tight_layout()
plt.show()