import numpy as np
import matplotlib.pyplot as plt

# Vector de tiempo
t = np.linspace(-1, 2, 1000)  # Ajusta el rango según sea necesario

# Señal Exponencial Decreciente (un tramo)
exp_decreciente = np.exp(-t) * ((t >= 0) & (t < 1))

# Señal Exponencial Creciente (un tramo)
exp_creciente = np.exp(t) * ((t >= 0) & (t < 1))

# Impulso de Dirac (aproximación)
impulso = np.where(t == 0, 1, 0)  # Aproximación básica para graficar

# Señal Escalón
escalon = np.where(t >= 0, 1, 0)

# Señal Sinc
x = np.linspace(-10, 10, 1000)
sinc = np.sinc(x / np.pi)  # np.sinc en numpy ya incluye pi (sinc(x/pi))

# Graficar las señales
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(t, exp_decreciente)
plt.title('Exponencial Decreciente (un tramo)')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, exp_creciente)
plt.title('Exponencial Creciente (un tramo)')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.stem(t, impulso)
plt.title('Impulso (Aproximado)')
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(t, escalon)
plt.title('Escalón')
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(x, sinc)
plt.title('Sinc(x)')
plt.grid(True)

plt.tight_layout()
plt.show()