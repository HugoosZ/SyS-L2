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


# Ondas Periodicas
senoidal = Am * np.sin(2 * np.pi * f * t)
cuadrada = Am * signal.square(2 * np.pi * f * t)
triangular = Am * signal.sawtooth(2 * np.pi * f * t, 0.5)
diente_sierra = Am * signal.sawtooth(2 * np.pi * f * t)

# Ondas A-periodicas
exp_decreciente = np.exp(-a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))
exp_creciente = np.exp(a * t) * ((np.heaviside(t, 1)) - (np.heaviside(t - 1, 1)))

index_cero = np.argmin(np.abs(t - 0))
impulso = np.zeros_like(t)
impulso[index_cero] = 1

escalon = np.heaviside(t, 1)

# Convoluciones
conv_seno_expon_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')[:len(t)]
conv_cuadrada_expon_creciente = np.convolve(cuadrada, exp_creciente, mode='full')[:len(t)]
conv_impulso_triangular = np.convolve(triangular, impulso, mode='same')
conv_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')[:len(t)]

# Funcion para calculo de energía 
def calcular_energia(signal):
    return np.sum(np.abs(signal) ** 2) 

# Intervalo de tiempo entre muestras
delta_t = t[1] - t[0]

# Funcion para calculo de potencia media 
def calcular_potencia(signal,delta_t):
    return np.mean((np.abs(signal) ** 2)/delta_t)



# Energía y potencia en las señales de entrada 
energia_senoidal = calcular_energia(senoidal)
potencia_senoidal = calcular_potencia(senoidal, delta_t)

energia_cuadrada = calcular_energia(cuadrada)
potencia_cuadrada = calcular_potencia(cuadrada, delta_t)

energia_triangular = calcular_energia(triangular)
potencia_triangular = calcular_potencia(triangular, delta_t)

energia_diente_sierra = calcular_energia(diente_sierra)
potencia_diente_sierra = calcular_potencia(diente_sierra, delta_t)



# Energía y potencia en las señales de salida 
energia_conv_seno = calcular_energia(conv_seno_expon_decreciente)
potencia_conv_seno = calcular_potencia(conv_seno_expon_decreciente, delta_t)

energia_conv_cuadrada = calcular_energia(conv_cuadrada_expon_creciente)
potencia_conv_cuadrada = calcular_potencia(conv_cuadrada_expon_creciente, delta_t)

energia_conv_triangular = calcular_energia(conv_impulso_triangular)
potencia_conv_triangular = calcular_potencia(conv_impulso_triangular, delta_t)

energia_conv_sierra = calcular_energia(conv_sierra_escalon)
potencia_conv_sierra = calcular_potencia(conv_sierra_escalon, delta_t)

# Resultados
print(f"Energía y potencia de las señales de entrada:")
print(f"Senoidal: Energía = {energia_senoidal:.4f}, Potencia media = {potencia_senoidal:.4f}")
print(f"Cuadrada: Energía = {energia_cuadrada:.4f}, Potencia media = {potencia_cuadrada:.4f}")
print(f"Triangular: Energía = {energia_triangular:.4f}, Potencia media = {potencia_triangular:.4f}")
print(f"Diente de sierra: Energía = {energia_diente_sierra:.4f}, Potencia media = {potencia_diente_sierra:.4f}")

print(f"\nEnergía y potencia de las señales de salida (convoluciones):")
print(f"Senoidal con Exponencial Decreciente: Energía = {energia_conv_seno:.4f}, Potencia media = {potencia_conv_seno:.4f}")
print(f"Cuadrada con Exponencial Creciente: Energía = {energia_conv_cuadrada:.4f}, Potencia media = {potencia_conv_cuadrada:.4f}")
print(f"Triangular con Impulso: Energía = {energia_conv_triangular:.4f}, Potencia media = {potencia_conv_triangular:.4f}")
print(f"Diente de Sierra con Escalón: Energía = {energia_conv_sierra:.4f}, Potencia media = {potencia_conv_sierra:.4f}")


