import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy import ndimage

import os
dir = os.path.dirname(os.path.realpath(__file__))

import tkinter as tk
from tkinter import filedialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

def normalize(data):
    data /= max(np.amax(data), abs(np.amin(data)))
    return data

def normalizeByDType(data):
    if(data.dtype == np.dtype('uint8')):
        bitcount = 8
    elif(data.dtype == np.dtype('int16')):
        bitcount = 16
    elif(data.dtype == np.dtype('int32')):
        bitcount = 32
    elif(data.dtype == np.dtype('float32')):
        bitcount = 1
    else:
        bitcount = 16
    data = (data / 2**(bitcount-1)) # normalize to -1 to 1
    return data

rate, data = wavfile.read(file_path)

data = normalizeByDType(data)
if(len(data.shape)>1):
    data = data[:,0] # make mono

plt.figure()
plt.subplot(2,2,1)
plt.plot(data)
plt.ylim([-1, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')


f, t, spect = signal.stft(data, window='tukey')
plt.subplot(2,2,2)
plt.pcolormesh(t/rate, f, np.abs(spect), shading='gouraud')
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')





# Do what you wish with the ndarray of spectral values
# Some filters as a starting point: https://docs.scipy.org/doc/scipy-1.8.0/html-scipyorg/reference/ndimage.html#module-scipy.ndimage

# Quick and dirty first step for some filters which can't take complex values
# spect = np.real(spect)

# Spectral noise gate
# spect = np.where(np.abs(spect) >= 1/100, spect, 0)

# Convolution
# weights = np.array([[0,10,0],[1,1,1],[0,0,0]])
# spect = ndimage.convolve(spect,weights)

# Dilation/Erosion
# spect = ndimage.grey_dilation(np.real(spect),1)
# spect = ndimage.grey_erosion(np.real(spect),10)
# spect = ndimage.grey_closing(np.real(spect),30)
# spect = ndimage.grey_opening(np.real(spect),30)

# Edge detection
spect = ndimage.sobel(spect)
# spect = np.hypot(ndimage.sobel(spect,0), ndimage.sobel(spect,1))

# Blurring
# spect = ndimage.gaussian_filter(spect,10)




plt.subplot(2,2,4)
plt.pcolormesh(t/rate, f, np.abs(spect), shading='gouraud')
plt.ylim([f[1], f[-1]])
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.yscale('log')

tout, dataout = signal.istft(spect)
dataout = normalize(dataout)

plt.subplot(2,2,3)
plt.plot(tout/rate,dataout)
plt.ylim([-1, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.show()

# make stereo and proper bit depth
dataout = np.asarray(dataout, dtype=np.float32)
dataout = np.stack([dataout,dataout],1)

wavfile.write(dir + '/' + 'output.wav', rate, dataout)

