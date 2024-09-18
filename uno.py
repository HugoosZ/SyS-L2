import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros comunes
fs = 1000  # Frecuencia de muestreo (Hz)
T = 1  # Duración en segundos
t = np.linspace(0, T, int(fs * T), endpoint=False)  # Vector de tiempo

# Señal Senoidal
freq = 5  # Frecuencia de la señal (Hz)
senoidal = np.sin(2 * np.pi * freq * t)

# Señal Cuadrada
cuadrada = signal.square(2 * np.pi * freq * t)

# Señal Triangular
triangular = signal.sawtooth(2 * np.pi * freq * t, 0.5)

# Señal Diente de Sierra
diente_sierra = signal.sawtooth(2 * np.pi * freq * t)

# Graficar las señales
plt.figure(figsize=(12, 8))

plt.subplot(4, 1, 1)
plt.plot(t, senoidal)
plt.title('Señal Senoidal')
plt.grid(True)

plt.subplot(4, 1, 2)
plt.plot(t, cuadrada)
plt.title('Señal Cuadrada')
plt.grid(True)

plt.subplot(4, 1, 3)
plt.plot(t, triangular)
plt.title('Señal Triangular')
plt.grid(True)

plt.subplot(4, 1, 4)
plt.plot(t, diente_sierra)
plt.title('Señal Diente de Sierra')
plt.grid(True)

plt.tight_layout()
plt.show()