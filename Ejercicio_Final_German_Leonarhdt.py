
# EJercicio final  - Germán Leonarhdt

print(__doc__)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
from io import StringIO


# Importamos bases Baseline y pestañeos

# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

print('Baseline - Estructura de la informacion:')
print(baseline.head())


# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

print('Pestañeos - Estructura de la informacion:')
print(pestaneos.head())


# Chequeamos nulos y duplicados

# Nulos

# Baseline
baseline_df = pd.DataFrame(baseline)
print('El dataset de Baseline tiene',len(baseline_df),'observaciones')
baseline_nonulls = baseline_df.dropna()
print('Y si removemos nulos ',len(baseline_nonulls),'observaciones, la misma cantidad')
baseline_null=baseline_df.isnull().sum().sum()
print('Es decir, el dataset de Baseline tiene',baseline_null,'nulos.')



baseline_nonulls = baseline.dropna()
print(baseline_nonulls)

print('The same as')

print(data[data.notnull()])




# Graficamos las series sin modificar

# Baseline
plt.plot(baseline_eeg,'r', label='EEG')
plt.xlabel('t');f
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal - Baseline')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(baseline_eeg)])
plt.savefig('images/Baseline_signal.png')
plt.show()

# Pestañeos
plt.plot(pestaneos_eeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'EEG Signal - Pestañeos')     # r'' representa un raw string que no tiene caracteres especiales
plt.ylim([-2000, 2000]);
plt.xlim([0,len(pestaneos_eeg)])
plt.savefig('images/Pestaneos_signal.png')
plt.show()


# Aplicamos filtro temporal

# Operacion de convulsion

# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]
windowlength = 10
baseline_avgeeg = np.convolve(baseline_eeg, np.ones((windowlength,))/windowlength, mode='same')
plt.plot(baseline_avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal - Baseline')     
plt.ylim([-2000, 2000]);
plt.xlim([0,len(baseline_avgeeg)])
plt.savefig('images/baseline_smoothed.png')
plt.show()

# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

windowlength = 10
pestaneos_avgeeg = np.convolve(pestaneos_eeg, np.ones((windowlength,))/windowlength, mode='same')
plt.plot(pestaneos_avgeeg,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Smoothed EEG Signal - Pestañeos')     
plt.ylim([-2000, 2000]);
plt.xlim([0,len(pestaneos_avgeeg)])
plt.savefig('images/pestaneos_smoothed.png')
plt.show()

# Operacion de normalizacion

def z_score_norm(arr):
    """Apply z-score normalization
        to an array or series
    """
    mean_ = np.mean(arr)
    std_ = np.std(arr)

    new_arr = [(i-mean_)/std_ for i in arr]

    return new_arr


# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

baseline_eeg_zscore = z_score_norm(baseline_eeg)

plt.plot(baseline_eeg_zscore,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Z-score EEG Signal- Baseline')     
plt.ylim([-20, 20]);
plt.xlim([0,len(baseline_eeg_zscore)])
plt.savefig('images/baseline_zscoredeeg.png')
plt.show()

# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

pestaneos_eeg_zscore = z_score_norm(pestaneos_eeg)

plt.plot(pestaneos_eeg_zscore,'r', label='EEG')
plt.xlabel('t');
plt.ylabel('eeg(t)');
plt.title(r'Z-score EEG Signal- Pestañeos')     
plt.ylim([-20, 20]);
plt.xlim([0,len(pestaneos_eeg_zscore)])
plt.savefig('images/pestaneos_zscoredeeg.png')
plt.show()


# Aplicamos filtro espectral


import sys, select

import time
import datetime
import os

from scipy.fftpack import fft

import math

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from scipy.signal import butter, lfilter



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def psd(y):
    # Number of samplepoints
    N = 512
    # sample spacing
    T = 1.0 / 512.0
 
    # Original Bandpass
    fs = 512.0
    fso2 = fs/2
 
    y = butter_bandpass_filter(y, 8.0, 15.0, fs, order=6)
    yf = fft(y)
   

    return np.sum(np.abs(yf[0:int(N/2)]))


Fs = 512.0

# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

baseline_normalized_signal = baseline_eeg

N_baseline = len(baseline_normalized_signal)


# Creo una secuencia de N puntos (el largo de EEG), de 0 hasta el largo de la secuencia en segundos (N/Fs).
x = np.linspace(0.0, int(N_baseline/Fs), N_baseline)   

# A esa secuencia de EEG le agrego una señal pura de 30 Hz.  Estoy ayuda a visualizar bien que la relación espectral está ok.
baseline_normalized_signal +=  100*np.sin(30.0 * 2.0*np.pi*x)

yf = rfft(baseline_normalized_signal)
xf = rfftfreq(N_baseline, 1 / Fs)

plt.figure(figsize=(14,7))
plt.title('Frequency Spectrum - Baseline')
plt.plot(xf, np.abs(yf), color='green')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hertz)')
plt.savefig('images/baseline_spectral.png')
plt.show()


# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

pestaneos_normalized_signal = pestaneos_eeg

N_pestaneos = len(pestaneos_normalized_signal)


# Creo una secuencia de N puntos (el largo de EEG), de 0 hasta el largo de la secuencia en segundos (N/Fs).
x = np.linspace(0.0, int(N_pestaneos/Fs), N_pestaneos)   

# A esa secuencia de EEG le agrego una señal pura de 30 Hz.  Estoy ayuda a visualizar bien que la relación espectral está ok.
pestaneos_normalized_signal +=  100*np.sin(30.0 * 2.0*np.pi*x)

yf = rfft(pestaneos_normalized_signal)
xf = rfftfreq(N_pestaneos, 1 / Fs)

plt.figure(figsize=(14,7))
plt.title('Frequency Spectrum - Pestañeos')
plt.plot(xf, np.abs(yf), color='green')
plt.ylabel('Amplitude')
plt.xlabel('Frequency (Hertz)')
plt.savefig('images/pestaneos_spectral.png')
plt.show()

# Filtro espacial
# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

from sklearn.decomposition import FastICA, PCA

S = np.c_[baseline_eeg, pestaneos_eeg]

ica = FastICA(n_components=2)
S_ = ica.fit_transform(S)  # Reconstruct signals
A_ = ica.mixing_  # Get estimated mixing matrix
assert np.allclose(S, np.dot(S_, A_.T) + ica.mean_)

plt.figure(4)
plt.title('ICA 1')
plt.subplot(2,1,1)
plt.plot(S_[:,0], color='red')
plt.title('ICA 2')
plt.subplot(2,1,2)
plt.plot(S_[:,1], color='steelblue')
plt.show()

# Signal Features

print(__doc__)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft

import math

from scipy.signal import firwin, remez, kaiser_atten, kaiser_beta
from scipy.signal import butter, filtfilt, buttord

from scipy.signal import butter, lfilter

import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def psd(y):
    # Number of samplepoints
    N = 512
    # sample spacing
    T = 1.0 / 512.0
    # From 0 to N, N*T, 2 points.
    #x = np.linspace(0.0, 1.0, N)
    #y = 1*np.sin(10.0 * 2.0*np.pi*x) + 9*np.sin(20.0 * 2.0*np.pi*x)


    # Original Bandpass
    fs = 512.0
    fso2 = fs/2
    #Nd,wn = buttord(wp=[9/fso2,11/fso2], ws=[8/fso2,12/fso2],
    #   gpass=3.0, gstop=40.0)
    #b,a = butter(Nd,wn,'band')
    #y = filtfilt(b,a,y)

    y = butter_bandpass_filter(y, 8.0, 15.0, fs, order=6)


    yf = fft(y)
    #xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))
    #import matplotlib.pyplot as plt
    #plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
    #plt.axis((0,60,0,1))
    #plt.grid()
    #plt.show()

    return np.sum(np.abs(yf[0:int(N/2)]))

def crest_factor(x):
    return np.max(np.abs(x))/np.sqrt(np.mean(np.square(x)))

def hjorth(a):
    r"""
    Compute Hjorth parameters [HJO70]_.
    .. math::
        Activity = m_0 = \sigma_{a}^2
    .. math::
        Complexity = m_2 = \sigma_{d}/ \sigma_{a}
    .. math::
        Morbidity = m_4 =  \frac{\sigma_{dd}/ \sigma_{d}}{m_2}
    Where:
    :math:`\sigma_{x}^2` is the mean power of a signal :math:`x`. That is, its variance, if it's mean is zero.
    :math:`a`, :math:`d` and :math:`dd` represent the original signal, its first and second derivatives, respectively.
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which appear to uses a non normalised (by the length of the signal) definition of the activity:
        .. math::
            \sigma_{a}^2 = \sum{\mathbf{x}[i]^2}
        As opposed to
        .. math::
            \sigma_{a}^2 = \frac{1}{n}\sum{\mathbf{x}[i]^2}
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: activity, complexity and morbidity
    :rtype: tuple(float, float, float)
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> activity, complexity, morbidity = pr.univariate.hjorth(noise)
    """

    first_deriv = np.diff(a)
    second_deriv = np.diff(a,2)

    var_zero = np.mean(a ** 2)
    var_d1 = np.mean(first_deriv ** 2)
    var_d2 = np.mean(second_deriv ** 2)

    activity = var_zero
    morbidity = np.sqrt(var_d1 / var_zero)
    complexity = np.sqrt(var_d2 / var_d1) / morbidity

    return activity, morbidity, complexity

def pfd(a):
    r"""
    Compute Petrosian Fractal Dimension of a time series [PET95]_.
    It is defined by:
    .. math::
        \frac{log(N)}{log(N) + log(\frac{N}{N+0.4N_{\delta}})}
    .. note::
        **Difference with PyEEG:**
        Results is different from [PYEEG]_ which implemented an apparently erroneous formulae:
        .. math::
            \frac{log(N)}{log(N) + log(\frac{N}{N}+0.4N_{\delta})}
    Where:
    :math:`N` is the length of the time series, and
    :math:`N_{\delta}` is the number of sign changes.
    :param a: a one dimensional floating-point array representing a time series.
    :type a: :class:`~numpy.ndarray` or :class:`~pyrem.time_series.Signal`
    :return: the Petrosian Fractal Dimension; a scalar.
    :rtype: float
    Example:
    >>> import pyrem as pr
    >>> import numpy as np
    >>> # generate white noise:
    >>> noise = np.random.normal(size=int(1e4))
    >>> pr.univariate.pdf(noise)
    """

    diff = np.diff(a)
    # x[i] * x[i-1] for i in t0 -> tmax
    prod = diff[1:-1] * diff[0:-2]

    # Number of sign changes in derivative of the signal
    N_delta = np.sum(prod < 0)
    n = len(a)

    return np.log(n)/(np.log(n)+np.log(n/(n+0.4*N_delta)))



# Sampling frequency of 512 Hz


# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

ptp = abs(np.max(baseline_eeg)) + abs(np.min(baseline_eeg))
rms = np.sqrt(np.mean(baseline_eeg**2))
cf = crest_factor(baseline_eeg)

print ('Peak-To-Peak:' + str(ptp))
print ('Root Mean Square:' + str(rms))
print ('Crest Factor:' + str(cf))

from collections import Counter
from scipy import stats

entropy = stats.entropy(list(Counter(baseline_eeg).values()), base=2)

print('Shannon Entropy:' + str(entropy))


activity, complexity, morbidity = hjorth(baseline_eeg)

print('Activity:' + str(activity))
print('Complexity:' + str(complexity))
print('Mobidity:' + str(morbidity))



fractal = pfd(baseline_eeg)
print('Fractal:' + str(fractal))

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

peaks, _ = find_peaks(eeg, height=200)
plt.plot(baseline_eeg)
plt.plot(peaks, baseline_eeg[peaks], "x")
plt.plot(np.zeros_like(baseline_eeg), "--", color="gray")
plt.show()


N = 512
T = 1.0 / 512.0

# We can put an additional frequency component to verify that things are working ok
shamsignal = False
if (shamsignal):
    x= np.linspace(0.0, 1.0, N)
    baseline_eeg = baseline_eeg[:512] +  100*np.sin(10.0 * 2.0*np.pi*x)


yf = fft(baseline_eeg)
xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))

plt.close()

plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.grid()
plt.show()

print('PSD:' + str(psd(baseline_eeg[:512])))




# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

ptp = abs(np.max(pestaneos_eeg)) + abs(np.min(pestaneos_eeg))
rms = np.sqrt(np.mean(pestaneos_eeg**2))
cf = crest_factor(pestaneos_eeg)

print ('Peak-To-Peak:' + str(ptp))
print ('Root Mean Square:' + str(rms))
print ('Crest Factor:' + str(cf))

from collections import Counter
from scipy import stats

entropy = stats.entropy(list(Counter(pestaneos_eeg).values()), base=2)

print('Shannon Entropy:' + str(entropy))


activity, complexity, morbidity = hjorth(pestaneos_eeg)

print('Activity:' + str(activity))
print('Complexity:' + str(complexity))
print('Mobidity:' + str(morbidity))



fractal = pfd(pestaneos_eeg)
print('Fractal:' + str(fractal))

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

peaks, _ = find_peaks(eeg, height=200)
plt.plot(pestaneos_eeg)
plt.plot(peaks, pestaneos_eeg[peaks], "x")
plt.plot(np.zeros_like(pestaneos_eeg), "--", color="gray")
plt.show()


N = 512
T = 1.0 / 512.0

# We can put an additional frequency component to verify that things are working ok
shamsignal = False
if (shamsignal):
    x= np.linspace(0.0, 1.0, N)
    pestaneos_eeg = pestaneos_eeg[:512] +  100*np.sin(10.0 * 2.0*np.pi*x)


yf = fft(pestaneos_eeg)
xf = np.linspace(0.0, int(1.0/(2.0*T)), int(N/2))

plt.close()

plt.plot(xf, 2.0/N * np.abs(yf[0:int(N/2)]))
plt.grid()
plt.show()

print('PSD:' + str(psd(pestaneos_eeg[:512])))



# Estadisticas descriptivas

# Baseline
baseline = pd.read_csv('data/baseline.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
baseline = baseline.values
baseline_eeg = baseline[:,2]

print('File Length:'+str(len(baseline)))
print("Some values from the dataset:\n")
print(results[0:10,])
print("Matrix dimension: {}".format(baseline.shape))
print("EEG Vector Metrics\n")
print("Length: {}".format(len(baseline_eeg)))
print("Max value: {}".format(baseline_eeg.max()))
print("Min value: {}".format(baseline_eeg.min()))
print("Range: {}".format(baseline_eeg.max()-baseline_eeg.min()))
print("Average value: {}".format(baseline_eeg.mean()))
print("Variance: {}".format(baseline_eeg.var()))
print("Std: {}".format(math.sqrt(baseline_eeg.var())))
plt.figure(figsize=(12,5))
plt.plot(baseline_eeg,color="green")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoints",size=10)
plt.title("Serie temporal de  - Baseline",size=20)
plt.show()

# Pestañeos
pestaneos = pd.read_csv('data/pestaneos.dat', delimiter=' ', names = ['timestamp','counter','eeg','attention','meditation','blinking'])
pestaneos = pestaneos.values
pestaneos_eeg = pestaneos[:,2]

print('File Length:'+str(len(pestaneos)))
print("Some values from the dataset:\n")
print(results[0:10,])
print("Matrix dimension: {}".format(pestaneos.shape))
print("EEG Vector Metrics\n")
print("Length: {}".format(len(pestaneos_eeg)))
print("Max value: {}".format(pestaneos_eeg.max()))
print("Min value: {}".format(pestaneos_eeg.min()))
print("Range: {}".format(pestaneos_eeg.max()-pestaneos_eeg.min()))
print("Average value: {}".format(pestaneos_eeg.mean()))
print("Variance: {}".format(pestaneos_eeg.var()))
print("Std: {}".format(math.sqrt(pestaneos_eeg.var())))
plt.figure(figsize=(12,5))
plt.plot(pestaneos_eeg,color="green")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoints",size=10)
plt.title("Serie temporal de  - pestaneos",size=20)
plt.show()


# Prueba de normalidad

# Baseline 
print('normality = {}'.format(scipy.stats.normaltest(baseline_eeg)))
sns.distplot(baseline_eeg)
plt.title("Normality-1 Analysis on EEG vector")
plt.show()
sns.boxplot(baseline_eeg,color="red")
plt.title("Normality-2 Analysis on EEG vector")
plt.show()
res = stats.probplot(baseline_eeg, plot = plt)
plt.title("Normality-3 Analysis on EEG vector") 
plt.show()

# Pestañeos 
print('normality = {}'.format(scipy.stats.normaltest(pestaneos_eeg)))
sns.distplot(pestaneos_eeg)
plt.title("Normality-1 Analysis on EEG vector")
plt.show()
sns.boxplot(pestaneos_eeg,color="red")
plt.title("Normality-2 Analysis on EEG vector")
plt.show()
res = stats.probplot(pestaneos_eeg, plot = plt)
plt.title("Normality-3 Analysis on EEG vector") 
plt.show()


# Aplico la metodologia de event counter para mi dataset de pestaneos

umbral_superior=int(pestaneos_eeg.mean()+3*pestaneos_eeg.std())
print("Upper Threshold: {}".format(umbral_superior))
umbral_inferior=int(pestaneos_eeg.mean()-3*pestaneos_eeg.std())
print("Lower Threshold: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(pestaneos_eeg,color="green")
plt.plot(np.full(len(pestaneos_eeg),umbral_superior),'r--')
plt.plot(np.full(len(pestaneos_eeg),umbral_inferior),'r--')
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.annotate("Upper Threshold",xy=(500,umbral_superior+10),color="red")
plt.annotate("Lower Threshold",xy=(500,umbral_inferior+10),color="red")
plt.show()


filtro_eeg=[]
contador=0
for i in range(len(pestaneos_eeg)):
    if i==0:
        filtro_eeg.append(0)
    elif pestaneos_eeg[i]>umbral_superior:
        filtro_eeg.append(1)
        if pestaneos_eeg[i-1]<=umbral_superior:
            print(i)
            contador=contador+1
    elif pestaneos_eeg[i]<umbral_inferior:
        filtro_eeg.append(-1)
    else:
        filtro_eeg.append(0)
        
print("Blinking counter: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.show()



#Find the threshold values to determine what is a blinking and what is not

# Como alternativa, genero los umbrales utilizando el dataset de baseline, y luego los aplico a pestaneos

# Defino el umbral usando el dataset de baseline
umbral_superior=int(baseline_eeg.mean()+3*baseline_eeg.std())
print("Upper Threshold: {}".format(umbral_superior))
umbral_inferior=int(baseline_eeg.mean()-3*baseline_eeg.std())
print("Lower Threshold: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(baseline_eeg,color="green")
plt.plot(np.full(len(baseline_eeg),umbral_superior),'r--')
plt.plot(np.full(len(baseline_eeg),umbral_inferior),'r--')
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.annotate("Upper Threshold",xy=(500,umbral_superior+10),color="red")
plt.annotate("Lower Threshold",xy=(500,umbral_inferior+10),color="red")
plt.show()


# Testeo si el umbral funciona para la prueba de pestañeos

#umbral_superior=int(pestaneos_eeg.mean()+3*pestaneos_eeg.std())
#print("Upper Threshold: {}".format(umbral_superior))
#umbral_inferior=int(pestaneos_eeg.mean()-3*pestaneos_eeg.std())
#print("Lower Threshold: {}".format(umbral_inferior))
plt.figure(figsize=(12,5))
plt.plot(pestaneos_eeg,color="green")
plt.plot(np.full(len(pestaneos_eeg),umbral_superior),'r--')
plt.plot(np.full(len(pestaneos_eeg),umbral_inferior),'r--')
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.annotate("Upper Threshold",xy=(500,umbral_superior+10),color="red")
plt.annotate("Lower Threshold",xy=(500,umbral_inferior+10),color="red")
plt.show()



filtro_eeg=[]
contador=0
for i in range(len(pestaneos_eeg)):
    if i==0:
        filtro_eeg.append(0)
    elif pestaneos_eeg[i]>umbral_superior:
        filtro_eeg.append(1)
        if pestaneos_eeg[i-1]<=umbral_superior:
            print(i)
            contador=contador+1
    elif pestaneos_eeg[i]<umbral_inferior:
        filtro_eeg.append(-1)
    else:
        filtro_eeg.append(0)
        
print("Blinking counter: {}".format(contador))
filtro_eeg=np.asarray(filtro_eeg)
plt.figure(figsize=(16,5))
plt.plot(filtro_eeg,color="blue")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.show()


# Pruebo el metodo alternativo que propone el creador del script Eventcounter.py, solo cambio el umbral de 420 a 600
# Alternative method

# The threshold is hardcoded, visually estimated.
signalthreshold = 600

eeg=pestaneos_eeg
# Filter the values above the threshold
boolpeaks = np.where( eeg > signalthreshold  )
print (boolpeaks)

# Pick the derivative
dpeaks = np.diff( eeg )
print (dpeaks)

# Identify those values where the derivative is ok
pdpeaks = np.where( dpeaks > 0)

peaksd = pdpeaks[0] 

# boolpeaks and peaksd are indexes.
finalresult = np.in1d(peaksd,boolpeaks)

print (finalresult)     
blinkings = finalresult.sum()

peaks1 = peaksd[finalresult]

print ('Blinkings: %d' % blinkings)
print ('Locations:');print(peaks1)

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

peaks2, _ = find_peaks(eeg, height=200)
plt.plot(eeg)
plt.plot(np.full(len(eeg),signalthreshold),'r--')
plt.plot(peaks2, eeg[peaks2], "x")
plt.plot(peaks1, eeg[peaks1], "o")
plt.plot(np.zeros_like(eeg), "--", color="gray")
plt.show()


# Tomo lo realizado en el scrip del Alumno: Francisco Seguí https://github.com/fseguior/

# Copy-paste
# Propongo una forma alternativa de delimitar dinámicamente los umbrales de detección de pestañeo
# Calculo los límites inferiores y superiores utilizando una medida de posición
# En este caso uso el percentil 1 y el 99, con lo cual se consideran como picos el 2% de los valores
# De esta forma el filtro es dinámico, y se adapta a los valores de la muestra.
# Vemos en el gráfico que el criterio funciona adecuadamente

lowerbound=int(np.percentile(eeg, 1))
upperbound=int(np.percentile(eeg, 99))

plt.plot(eeg, color="steelblue")
plt.plot(np.full(len(eeg),lowerbound), color="goldenrod", ls="--")
plt.plot(np.full(len(eeg),upperbound), color="goldenrod", ls="--")
plt.ylabel("Amplitude",size=10)
plt.xlabel("Timepoint",size=10)
plt.title("EEG Series with control limits",size=20)
plt.ylim([min(eeg)*1.1, max(eeg)*1.1 ])  ## dinamizo los valores del eje así se adapta a los datos que proceso
plt.annotate("Lower Bound",xy=(500,lowerbound+10),color="goldenrod")
plt.annotate("Upper Bound",xy=(500,upperbound+10),color="goldenrod")
plt.savefig('blinks.png')
plt.show()

# Grafico el filtro de pestañeos/blinking
# Utilizo una función lambda para marcar los pestañeos

blinks = list((map(lambda x: 1 if x >upperbound else ( -1 if x < lowerbound else 0), eeg)))
blinks = np.asarray(blinks)

plt.plot(blinks, color="darksalmon")
plt.title("Blinking Filter",size=20)
plt.ylabel("Class",size=10)
plt.xlabel("Timepoint",size=10)
plt.savefig('blinkingfilter.png')
plt.show()

# Encuentro picos positivos. Filtro los valores donde blink==1, y luego analizo que haya habido un salto realmente (para no contar dos veces puntos consecutivos).
# Con un map y una funcion lambda obtengo una lista con booleanos para los valores donde hay picos realmente.
# Luego los filtro con una función filter y otra lambda
peak=np.where(blinks == 1)[0]

peakdiff=np.diff(np.append(0,peak))

boolpeak=list(map(lambda x : x > 100, peakdiff))

peakslocation=list(filter(lambda x: x, boolpeak*peak))

# Repito para los valles, mismo algoritmo pero busco blinks == -1
valley=np.where(blinks == -1)[0]

valleydiff=np.diff(np.append(0,valley))

boolvalley=list(map(lambda x : x > 100, valleydiff))

valleylocation=list(filter(lambda x: x, boolvalley*valley))

# Hago un append de los valles y los picos, y los ordeno. Luego los cuento para imprimir tanto la cantidad de pestañeos, como la localización de los mismos

blinklocations=np.sort(np.append(peakslocation,valleylocation))

blinkcount=np.count_nonzero(blinklocations)

print(f'Count of Blinks: {blinkcount}')
print('Location of Blinks');print(blinklocations)


