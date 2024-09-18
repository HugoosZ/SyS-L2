import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Vector de tiempo para las señales periódicas y aperiódicas
fs = 1000  # Frecuencia de muestreo
T = 1  # Duración en segundos
t_periodica = np.linspace(0, T, int(fs * T), endpoint=False)
t_aperiodica = np.linspace(-1, 2, int(fs * 3), endpoint=False)

# Señales Periódicas
senoidal = np.sin(2 * np.pi * 5 * t_periodica)  # Señal senoidal
cuadrada = signal.square(2 * np.pi * 5 * t_periodica)  # Señal cuadrada
triangular = signal.sawtooth(2 * np.pi * 5 * t_periodica, 0.5)  # Señal triangular
diente_sierra = signal.sawtooth(2 * np.pi * 5 * t_periodica)  # Señal diente de sierra

# Señales Aperiódicas (respuestas a impulso de los sistemas)
exp_decreciente = np.exp(-t_aperiodica) * ((t_aperiodica >= 0) & (t_aperiodica < 1))
exp_creciente = np.exp(t_aperiodica) * ((t_aperiodica >= 0) & (t_aperiodica < 1))
impulso = np.where(t_aperiodica == 0, 1, 0)  # Aproximación del impulso
escalon = np.where(t_aperiodica >= 0, 1, 0)
x = np.linspace(-10, 10, 1000)
sinc = np.sinc(x / np.pi)  # np.sinc normalizado

# Realizar convoluciones (cada señal periódica con una aperiódica)
conv_senoidal_exp_decreciente = np.convolve(senoidal, exp_decreciente, mode='same')[:len(t_periodica)]
conv_cuadrada_exp_creciente = np.convolve(cuadrada, exp_creciente, mode='same')[:len(t_periodica)]
conv_triangular_impulso = np.convolve(triangular, impulso, mode='same')[:len(t_periodica)]
conv_diente_sierra_escalon = np.convolve(diente_sierra, escalon, mode='same')[:len(t_periodica)]

# Graficar las señales convolucionadas
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.plot(t_periodica, conv_senoidal_exp_decreciente)
plt.title('Convolución: Señal Senoidal con Exponencial Decreciente')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t_periodica, conv_cuadrada_exp_creciente)
plt.title('Convolución: Señal Cuadrada con Exponencial Creciente')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t_periodica, conv_triangular_impulso)
plt.title('Convolución: Señal Triangular con Impulso')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t_periodica, conv_diente_sierra_escalon)
plt.title('Convolución: Señal Diente de Sierra con Escalón')
plt.grid(True)

plt.tight_layout()
plt.show()