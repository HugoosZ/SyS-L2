import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros comunes
Nm = 4000 # Número de muestras
f = 5       # Frecuencia de la señal (Hz)
T = 1  # Duración en segundos
t = np.linspace(-2, 2, Nm)  # Vector de tiempo
# Señal Senoidal
senoidal = np.sin(2 * np.pi * f * t)

# Señal Cuadrada
cuadrada = signal.square(2 * np.pi * 1 * t)

# Señal Triangular
triangular = signal.sawtooth(2 * np.pi * f * t, 0.5)

# Señal Diente de Sierra
diente_sierra = signal.sawtooth(2 * np.pi * f * t)

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