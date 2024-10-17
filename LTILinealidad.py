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


C = 2  # Factores de proporcionalidad
combinacion0 = C * senoidal + C * triangular
combinacion1 = C * cuadrada + C * diente_sierra
combinacion2 = C * senoidal + C * cuadrada
combinacion3 = C * triangular + C * diente_sierra
combinacion4 = C * senoidal + C * diente_sierra

# Convolución de la suma de señales 
conv_combinacion0 = np.convolve(combinacion0, exp_decreciente, mode='full')[:len(t)]
conv_combinacion1 = np.convolve(combinacion1, exp_creciente, mode='full')[:len(t)]
conv_combinacion2 = np.convolve(combinacion2, impulso, mode='same')
conv_combinacion3 = np.convolve(combinacion3, escalon, mode='full')[:len(t)]
conv_combinacion4 = np.convolve(combinacion4, sinc, mode='full')[:len(t)]


# Convoluciones individuales de cada señal
conv_seno0 = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_triangular0 = np.convolve(triangular, exp_decreciente, mode='full')[:len(t)]

conv_cuadrada1 = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_diente_sierra1 = np.convolve(diente_sierra, exp_creciente, mode='full')[:len(t)]

conv_seno2 = np.convolve(senoidal, impulso, mode='same')
conv_cuadrada2 = np.convolve(cuadrada, impulso, mode='same')

conv_triangular3 = np.convolve(triangular, escalon, mode='full')[:len(t)]
conv_diente_sierra3 = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

conv_seno4 = np.convolve(senoidal, sinc, mode='full')[:len(t)]
conv_diente_sierra4 = np.convolve(diente_sierra, sinc, mode='full')[:len(t)]


# Combinación lineal de las convoluciones individuales
conv_linealidad0 = C * conv_seno0 + C * conv_triangular0
conv_linealidad1 = C * conv_cuadrada1 + C * conv_diente_sierra1
conv_linealidad2 = C * conv_seno2 + C * conv_cuadrada2
conv_linealidad3 = C * conv_triangular3 + C * conv_diente_sierra3
conv_linealidad4 = C * conv_seno4 + C * conv_diente_sierra4

# Graficos
plt.figure(figsize=(12, 6))

plt.subplot(3, 2, 1)
plt.plot(t, conv_linealidad0, label='Convolucion-suma')
plt.plot(t, conv_combinacion0, '--', label='Suma-convolucion')
plt.title(f'Prueba de Linealidad senoidal y triangular en exponencial decreciente')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(t, conv_linealidad1, label='Convolucion-suma')
plt.plot(t, conv_combinacion1, '--', label='Suma-convolucion')
plt.title(f'Prueba de Linealidad cuadrada y diente de sierra en exponencial creciente')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(t, conv_linealidad2, label='Convolucion-suma')
plt.plot(t, conv_combinacion2, '--', label='Suma-convolucion')
plt.title(f'Prueba de Linealidad senoidal y cuadrada en impulso')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(t, conv_linealidad3, label='Convolucion-suma')
plt.plot(t, conv_combinacion3, '--', label='Suma-convolucion')
plt.title(f'Prueba de Linealidad triangular y diente de sierra en escalón')
plt.grid(True)
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(t, conv_linealidad4, label='Convolucion-suma')
plt.plot(t, conv_combinacion4, '--', label='Suma-convolucion')
plt.title(f'Prueba de Linealidad senoidal y diente de sierra en sinc')
plt.grid(True)
plt.legend()





plt.tight_layout()
plt.show()

