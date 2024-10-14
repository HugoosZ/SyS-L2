import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parámetros
f = 5  # Frecuencia de la onda
Am = 1  # Amplitud de la onda
Ti = -2  # Tiempo inicial
Tf = 2  # Tiempo final
Nm = 1000  # Número de muestras
t = np.linspace(Ti, Tf, Nm)
a = 1  # Factor de proporcionalidad de función exponencial
Desplazamiento = 1  # Corrimiento temporal (en segundos)
t_aplicado = t - Desplazamiento  # Aplicar corrimiento temporal hacia la derecha

# Ondas Periodicas
senoidal = Am * np.sin(2 * np.pi * f * t)
triangular = signal.sawtooth(2 * np.pi * f * t, 0.5)
cuadrada = Am * signal.square(2 * np.pi * f * t)
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)

# Ondas A-periodicas
exp_decreciente = np.exp(-a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))
exp_creciente = np.exp(a*t) * ((np.heaviside(t, 1)) - (np.heaviside(t-1, 1)))
cero = np.argmin(np.abs(t- 0))
impulso = np.zeros_like(t)
impulso[cero] = 1
escalon = np.heaviside(t, 1)
x = np.linspace(-10, 10, 1000)
sinc = np.sinc(x / np.pi)  

# Invariancia 
Desplazamiento = 1

# Respuestas al impulso
conv0 = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv1 = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv2 = np.convolve(triangular, impulso, mode='same')
conv3 = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]
conv4 = np.convolve(senoidal, sinc, mode='full')[:len(t)]

# Señal desplazada en el tiempo
senoidalD = Am * np.sin((2 * np.pi * f * t_aplicado))  # Señal desplazada en el tiempo
cuadradaD = signal.square((2 * np.pi * f * t_aplicado))
triangularD = signal.sawtooth((2 * np.pi * f * t_aplicado), 0.5)
diente_sierraD = signal.sawtooth(2 * np.pi * f * t_aplicado)

# Convolución de la señal desplazada 
convD0 = np.convolve(senoidalD, exp_decreciente, mode='full')[:len(t)]
convD1 = np.convolve(cuadradaD, exp_creciente, mode='full')[:len(t)]
convD2 = np.convolve(triangularD, impulso, mode='same')
convD3 = np.convolve(diente_sierraD, escalon, mode='full')[:len(t)]
convD4 = np.convolve(senoidalD, sinc, mode='full')[:len(t)]


# Graficos
plt.figure(figsize=(16, 10))


plt.subplot(5, 2, 1)
plt.plot(t, senoidal, label='Pre-Convolución')
plt.plot(t, conv0, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y triangular en exponencial decreciente sin desplazar' )
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 2)
plt.plot(t_aplicado, senoidalD, label='Pre-Convolución')
plt.plot(t_aplicado, convD0, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y triangular en exponencial decreciente desplazada')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 3)
plt.plot(t, cuadrada, label='Pre-Convolución')
plt.plot(t, conv1, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y triangular en exponencial decreciente')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 4)
plt.plot(t, cuadradaD, label='Pre-Convolución')
plt.plot(t_aplicado, convD1, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad cuadrada y diente de sierra en exponencial creciente')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 5)
plt.plot(t, triangular, label='Pre-Convolución')
plt.plot(t, conv2, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y cuadrada en impulso')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 6)
plt.plot(t, triangularD, label='Pre-Convolución')
plt.plot(t_aplicado, convD2, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y cuadrada en impulso')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 7)
plt.plot(t, diente_sierra, label='Pre-Convolución')
plt.plot(t, conv3, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad triangular y diente de sierra en escalón')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 8)
plt.plot(t, diente_sierraD, label='Pre-Convolución')
plt.plot(t_aplicado, convD3, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad triangular y diente de sierra en escalón')
plt.grid(True)
plt.legend()

plt.subplot(5, 2, 9)
plt.plot(t, senoidal, label='Pre-Convolución')
plt.plot(t, conv4, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y diente de sierra en sinc')
plt.grid(True)
plt.legend()


plt.subplot(5, 2, 10)
plt.plot(t, senoidalD, label='Pre-Convolución')
plt.plot(t_aplicado, convD4, '--', label='Post-Convolución')
plt.title(f'Prueba de Linealidad senoidal y diente de sierra en sinc')
plt.grid(True)
plt.legend()





plt.tight_layout()
plt.show()


# y[n - t0] = x[n] * h[n] = y[n-t0] = x[n - t0] * h[n] = 