# ============================================================
# Politécnica de Santa Rosa
#
# Materia: Aprendizaje automático
# Profesor: Jesús Salvador López Ortega
# Grupo: IRC02
# Archivo: scipy.py
# Descripción: Ejercicios básicos de manejo de scipy
# ============================================================
import numpy as np
from scipy import linalg, stats, optimize, signal
from typing import Callable
################################################################################
# NOTE: Revisa la API de SciPy en https://docs.scipy.org/doc//scipy/index.html #
################################################################################

# Ejercicio 1
# 
# TODO: Crea una función "solve_linear" que devuelva la solución a un sistema lineal Ax = b. 
# NOTE: https://docs.scipy.org/doc/scipy/reference/linalg.html
def solve_linear(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve un sistema lineal Ax = b.

    Parámetros:
    - A: np.ndarray
        Matriz de coeficientes (n x n).
    - b: np.ndarray
        Vector de términos independientes (n,).

    Retorna:
    - np.ndarray
        Solución del sistema lineal como vector columna.
    """
    solution = np.linalg.solve(A, b)
    return solution

# Ejercicio 2
#
# TODO: Crea una función "get_matrix_properties" que obtenga el determinante y la inversa en un tuple(det, inv) de una matriz de numpy. 
# NOTE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.det.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.inv.html
def get_matrix_properties(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el determinante y la inversa de una matriz.

    Parámetros:
    - mat: np.ndarray
        Matriz cuadrada (n x n).

    Retorna:
    - tuple: (det, inv)
        Determinante y matriz inversa.
    """
    det = np.linalg.det(mat)
    inv = np.linalg.inv(mat)
    return (det, inv)

# Ejercicio 3
#
# TODO: Crea una función "get_statistics" que devuelva la media, desviación estándar y moda en un tuple(mean, tstd, mode) sobre un arreglo de numpy.
# NOTE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.Binomial.mean.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.tstd.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mode.html
def get_statistics(arg: np.ndarray) -> tuple[float, float, float]:
    """
    Calcula la media, desviación estándar muestral y moda de un arreglo.

    Parámetros:
    - arg: np.ndarray
        Arreglo unidimensional de datos numéricos.

    Retorna:
    - tuple: (mean, tstd, mode)
        Media, desviación estándar y moda como flotantes.
    """
    mean =  float(np.mean(arg))
    tstd = float(np.std(arg, ddof=1))
    mode = float(stats.mode(arg, keepdims=True)[0][0])
    return (mean, tstd, mode)

# Ejercicio 4
# 
# TODO: Crea una función "find_min" que encuentre el minimo de una función y devuelva un objeto de tipo optimize.OptimizeResult
# NOTE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize_scalar.html
def find_min(fun: Callable[[float], float]) -> optimize.OptimizeResult:
    """
    Encuentra el mínimo de una función escalar.

    Parámetros:
    - fun: Callable[[float], float]
        Función objetivo a minimizar.

    Retorna:
    - OptimizeResult
        Objeto con los resultados de la optimización.
    """
    found_min = optimize.minimize_scalar(fun) 
    return found_min

# Ejercicio 5
#
# TODO: Crea una función "get_spectrum" que devuelva el espectro de una señal compuesta
# NOTE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.fft.fft.html
def get_spectrum(signal_pass: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula el espectro de una señal compuesta usando FFT.

    Parámetros:
    - signal: np.ndarray
        Señal en el dominio del tiempo.
    - sample_rate: float
        Frecuencia de muestreo en Hz.

    Retorna:
    - tuple: (frecuencias, magnitudes)
        Frecuencias positivas y sus magnitudes correspondientes.
    """
    N = len(signal_pass)
    
    # Calcular frecuencias (solo positivas)
    frequencies = np.fft.fftfreq(N, 1 / sample_rate)
    positive_indices = frequencies >= 0
    frequencies_positive = frequencies[positive_indices]
    K = len(frequencies_positive)
    
    # Pre-calcular ángulos para optimización
    magnitudes = np.zeros(K)
    
    for k in range(K):
        freq = frequencies_positive[k]
        real_sum = 0.0
        imag_sum = 0.0
        
        for n in range(N):
            angle = 2 * np.pi * freq * n / sample_rate
            real_sum += signal_pass[n] * np.cos(angle)
            imag_sum -= signal_pass[n] * np.sin(angle)  # Negativo por convención
        
        magnitudes[k] = np.sqrt(real_sum**2 + imag_sum**2) / N
    
    return (frequencies_positive, magnitudes)

# Ejercicio 6
#
# TODO: Crea una función "low_pass_filter" que aplique un filtro pasa bajas Butterworth sobre una señal con ruido. 
# NOTE: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
#       https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html
def low_pass_filter(signal_data: np.ndarray, fs: float, cutoff: float = 10.0, order: int = 4) -> np.ndarray:
    """
    Aplica un filtro pasa-bajas Butterworth a una señal con ruido.

    Parámetros:
    - signal: np.ndarray
        Señal de entrada en el dominio del tiempo.
    - fs: float
        Frecuencia de muestreo en Hz.
    - cutoff: float
        Frecuencia de corte del filtro en Hz (por defecto 10 Hz).
    - order: int
        Orden del filtro Butterworth (por defecto 4).

    Retorna:
    - np.ndarray
        Señal filtrada.
    """
    nyquist_freq = 0.5 * fs
    
    # Normaliza la frecuencia de corte (rango 0-1, donde 1 = Nyquist)
    normal_cutoff = cutoff / nyquist_freq
    
    # Diseña el filtro Butterworth (coeficientes a y b)
    b, a = signal.butter(N=order, Wn=normal_cutoff, btype='low', analog=False)
    
    # Filtro en ambas direcciones para eliminar el desfase
    clean_signal = signal.filtfilt(b, a, signal_data)
    return clean_signal