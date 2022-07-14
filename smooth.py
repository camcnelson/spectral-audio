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
plt.subplot(1,2,1)
plt.plot(data)
plt.ylim([-1, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')

x = 12
dataout = np.zeros(data.shape)
for i in range(x):
    dataout += (1/x)*np.roll(data,i*36,0)
dataout = normalize(dataout)

# dataout = np.zeros(data.shape)
# dataout[0::2] = data[1::2]
# dataout[1::2] = -data[0::2]

# dataout = data**3

plt.subplot(1,2,2)
plt.plot(dataout)
plt.ylim([-1, 1])
plt.xlabel('Time [sec]')
plt.ylabel('Signal')
plt.show()

# make stereo and proper bit depth
dataout = np.asarray(dataout, dtype=np.float32)
dataout = np.stack([dataout,dataout],1)

wavfile.write(dir + '/' + 'output.wav', rate, dataout)

