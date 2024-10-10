import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Vector de tiempo para las señales periódicas y aperiódicas
fs = 1000  # Frecuencia de muestreo
T = 1  # Duración en segundos
t_periodica = np.linspace(0, T, fs , endpoint=False)  # Vector de tiempo para señales periódicas
t_aperiodicae = np.linspace(0, T, fs , endpoint=False)  # Vector de tiempo para señales aperiódicas

# Señales Periódicas
senoidal = np.sin(2 * np.pi * 5 * t_periodica)  
cuadrada = signal.square(2 * np.pi * 5 * t_periodica)  
triangular = signal.sawtooth(2 * np.pi * 5 * t_periodica, 0.5)  
diente_sierra = signal.sawtooth(2 * np.pi * 5 * t_periodica)  

# Señales Aperiódicas (respuestas a impulso de los sistemas)
exp_decreciente = np.exp(-t_aperiodicae) * (np.heaviside(t_aperiodicae, 1) - np.heaviside(t_aperiodicae - 1, 1))
exp_creciente = np.exp(t_aperiodicae) * (np.heaviside(t_aperiodicae, 1) - np.heaviside(t_aperiodicae - 1, 1))

def ddf(t,sig):
    val = np.zeros_like(t)
    val[(-(1/(2*sig))<=t) & (t<=(1/(2*sig)))] = 1
    return val

sig=1000
impulso = ddf(t_aperiodicae,sig)
escalon = np.heaviside(t_aperiodicae, 1)

# Realizar convoluciones (todas en modo 'full')
conv_senoidal_exp_decreciente = np.convolve(senoidal, exp_decreciente, mode='full')
conv_cuadrada_exp_creciente = np.convolve(cuadrada, exp_creciente, mode='full')
conv_triangular_impulso = np.convolve(triangular, impulso, mode='full')
conv_diente_sierra_escalon = np.convolve(diente_sierra, escalon, mode='full')

def convolucion_manual(x, h):
    N = len(x)
    M = len(h)
    y = np.zeros(N + M - 1)
    
    # Invertir h
    h_inv = h[::-1]
    
    # Desplazar h y realizar el producto punto a punto
    for n in range(len(y)):
        suma = 0
        for k in range(M):
            if 0 <= n - k < N:
                suma += x[n - k] * h_inv[k]
        y[n] = suma
    
    return y

conv_senoidal_exp_decreciente_manual = convolucion_manual(senoidal, exp_decreciente)
conv_cuadrada_exp_creciente_manual = convolucion_manual(cuadrada, exp_creciente)
conv_triangular_impulso_manual = convolucion_manual(triangular, impulso)
conv_diente_sierra_escalon_manual = convolucion_manual(diente_sierra, escalon)

# Crear nuevos vectores de tiempo ajustados para las convoluciones en modo 'full'
t_full_senoidal = np.linspace(-5, T + 3 + len(exp_decreciente)/fs, len(conv_senoidal_exp_decreciente))
t_full_cuadrada = np.linspace(-5, T + 3 + len(exp_creciente)/fs, len(conv_cuadrada_exp_creciente))
t_full_triangular = np.linspace(-5, T + 3 + len(impulso)/fs, len(conv_triangular_impulso))
t_full_diente_sierra = np.linspace(-5, T + 3 + len(escalon)/fs, len(conv_diente_sierra_escalon))

# Graficar las señales periódicas, aperiódicas y convoluciones
plt.figure(figsize=(16, 10))

# Señales periódicas
plt.subplot(4, 4, 1)
plt.plot(t_periodica, senoidal)
plt.title('Señal Senoidal')

plt.subplot(4, 4, 2)
plt.plot(t_periodica, cuadrada)
plt.title('Señal Cuadrada')

plt.subplot(4, 4, 3)
plt.plot(t_periodica, triangular)
plt.title('Señal Triangular')

plt.subplot(4, 4, 4)
plt.plot(t_periodica, diente_sierra)
plt.title('Señal Diente de Sierra')

# Señales aperiódicas
plt.subplot(4, 4, 5)
plt.plot(t_aperiodicae, exp_decreciente)
plt.title('Exponencial Decreciente')

plt.subplot(4, 4, 6)
plt.plot(t_aperiodicae, exp_creciente)
plt.title('Exponencial Creciente')

plt.subplot(4, 4, 7)
plt.plot(t_aperiodicae, impulso)
plt.title('Impulso')

plt.subplot(4, 4, 8)
plt.plot(t_aperiodicae, escalon)
plt.title('Escalón')

# Convoluciones automáticas
plt.subplot(4, 4, 9)
plt.plot(t_full_senoidal, conv_senoidal_exp_decreciente)
plt.title('Conv: Senoidal con Exp. Decreciente (Full)')

plt.subplot(4, 4, 10)
plt.plot(t_full_cuadrada, conv_cuadrada_exp_creciente)
plt.title('Conv: Cuadrada con Exp. Creciente (Full)')

plt.subplot(4, 4, 11)
plt.plot(t_full_triangular, conv_triangular_impulso)
plt.title('Conv: Triangular con Impulso (Full)')

plt.subplot(4, 4, 12)
plt.plot(t_full_diente_sierra, conv_diente_sierra_escalon)
plt.title('Conv: Diente de Sierra con Escalón (Full)')

# Convoluciones manuales
plt.subplot(4, 4, 13)
plt.plot(t_full_senoidal, conv_senoidal_exp_decreciente_manual)
plt.title('Conv Manual: Senoidal con Exp. Decreciente')

plt.subplot(4, 4, 14)
plt.plot(t_full_cuadrada, conv_cuadrada_exp_creciente_manual)
plt.title('Conv Manual: Cuadrada con Exp. Creciente')

plt.subplot(4, 4, 15)
plt.plot(t_full_triangular, conv_triangular_impulso_manual)
plt.title('Conv Manual: Triangular con Impulso')

plt.subplot(4, 4, 16)
plt.plot(t_full_diente_sierra, conv_diente_sierra_escalon_manual)
plt.title('Conv Manual: Diente de Sierra con Escalón')

plt.tight_layout()
plt.show()