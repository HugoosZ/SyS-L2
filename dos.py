import numpy as np
import matplotlib.pyplot as plt

# Vector de tiempo
t = np.linspace(-1, 2, 10000) 

# Señal Exponencial Decreciente acotada
exp_decreciente = np.exp(-t) * (np.heaviside(t, 1) - np.heaviside(t - 1, 1))

# Señal Exponencial Creciente acotada
exp_creciente = np.exp(t) * (np.heaviside(t, 1) - np.heaviside(t - 1, 1))

# Impulso de Dirac 
index_cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[index_cero] = 1

# Señal Escalón
escalon = np.heaviside(t, 1)

# Señal Sinc
x = np.linspace(-10, 10, 1000)
sinc = np.sinc(x / np.pi)  

# Graficar las señales
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(t, exp_decreciente)
plt.title('Exponencial Decreciente acotada')
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(t, exp_creciente)
plt.title('Exponencial Creciente acotada')
plt.grid(True)

plt.subplot(5, 1, 3)
plt.stem(t, impulso)
plt.title('Impulso')
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