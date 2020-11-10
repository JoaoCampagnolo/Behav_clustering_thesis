# File:             findwavelets.py
# Date:             March 2019
# Description:      Finds the wavelet transforms resulting from a time series.
# Authors:          Joao Campagnolo
# Python version:   Python 3.7+

# Import packages:
import math
import numpy as np


def find_wav(modes, chan=25, omega0=5, fps=100, fmin=1, fmax=50):
    
    dt = 1/fps
    min_t = 1/fmax
    max_t = 1/fmin
    ts  = min_t * (2 ** (np.linspace(0, chan-1, chan) * math.log(max_t / min_t) / (math.log(2) * (chan-1))))
    f = 1/ts; #f = f[::-1]
    
    nframes = np.shape(modes)[0]
    nmodes = np.shape(modes)[1] 
    amplitudes = np.zeros(shape=(nframes, nmodes*chan))
    for i in range(nmodes):
        amplitudes[:,np.arange(chan)+(i-1)*chan] = np.transpose(morlet_wavelet_convolution(modes[:,i].copy(), f, omega0, dt))
        
    return amplitudes, f
        
        
def morlet_wavelet_convolution(x, f, omega0, dt):
    # finds the Morlet wavelet transform resulting from a 1d time-series.
    nframes = len(x)
    chan = len(f)
    amp = np.zeros(shape=(chan, nframes))
    if nframes % 2 == 1:
        x = np.append(x,0)
        nframes += 1
        odd = True
    else:
        odd = False
    
    x = np.pad(x, (nframes//2,nframes//2), 'constant', constant_values=(0,0))
    M = nframes;
    nframes = len(x)
    
    scales = (omega0 + math.sqrt(2+omega0**2))/(4*math.pi*f)
    omega_val = 2*math.pi*np.arange(-nframes/2, nframes/2, 1) / (nframes*dt)
    
    xHat = np.fft.fft(x) #fft(x)
    xHat = np.fft.fftshift(xHat) #fftshift(xHat)
    
    v = np.arange(0, nframes, 1)
    if odd:
        idx = np.isin(v, np.arange(M/2, M/2+M-1, 1)) # M/2+1 ??
#         idx = np.arange(M/2+1, M/2+M-1, 1)
    else:
        idx = np.isin(v, np.arange(M/2, M/2+M, 1))
#         idx = np.arange(M/2+1, M/2+M, 1)
    
    # parallel computing here
    # from multiprocessing import Pool
    for i in range(chan):
        mm = morletConjFT(-omega_val*scales[i], omega0)
        t1 = np.fft.ifft(mm*xHat)
        q = t1 * math.sqrt(scales[i])
        
        q = q[idx]
        
        amp[i,:] = abs(q) * (math.pi**(-.25)) * math.exp(.25*(omega0-math.sqrt(2+omega0**2))**2) / math.sqrt(2*scales[i])
        
    return amp

def morletConjFT(w, omega0):
    mor_transf = np.zeros(len(w))
    for i in range(len(w)):
        mor_transf[i] = math.pi**(-.25) * math.exp(-.5*(w[i]-omega0)**2)
    
    return mor_transf