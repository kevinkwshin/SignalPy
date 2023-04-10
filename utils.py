import torch
import torchaudio
import pylab as plt
import numpy as np

print(torch.__version__)
print(torchaudio.__version__)
print(np.__version__)
print(librosa.__version__)

import glob
import natsort
import numpy as np

import librosa.display
from scipy.signal import butter, lfilter, filtfilt
import torchaudio

from IPython.display import Audio

def load_data(path):
    x = np.load(path)
    x = np.expand_dims(np.stack([x,x],0),0)
    x = torch.from_numpy(x)
    m = torch.mean(x)
    s = torch.std(x)
    x = (x-m)/(s+1e-8)
    x = x.float()
    # x = torch.rand(1,2,4410*10)
    t = np.linspace(0,x.shape[-1]/sr,x.shape[-1])
    return x, t

def load_wav(path):
    x, sr = librosa.load(path, sr=11025)
    x = np.expand_dims(np.stack([x,x],0),0)
    x = torch.from_numpy(x)
    m = torch.mean(x)
    s = torch.std(x)
    x = (x-m)/(s+1e-8)
    x = x.float()
    # x = torch.rand(1,2,4410*10)
    t = np.linspace(0,x.shape[-1]/sr,x.shape[-1])
    return x, t

def draw_fft(x, sr=4000):
    try:
        x = x.numpy()
    except:
        pass
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='hz', )
    plt.colorbar(format="%+2.f dB")
    plt.ylim(0,2000)
    plt.show()

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def butter_highpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    y = filtfilt(b, a, data)
    return y

def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.ylim(0,10000)
    plt.show(block=False)
    
def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1, figsize=(20,6))
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show(block=False)

def draw_fft_plot(signal, sample_rate,label='',show=False):
    # Compute the FFT of the signal
    if len(signal.shape)==1:
        fft = torch.fft.fft(signal)
    elif len(signal.shape)==2:
        # get only 1st channel
        signal = signal[0]
        fft = torch.fft.fft(signal)
        
    # Compute the frequencies for the x-axis
    freqs = torch.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Plot the magnitude spectrum
    plt.magnitude_spectrum(signal, Fs=sample_rate,scale='dB', alpha=0.5, label=label)
    plt.title('FFT Plot')
    plt.xlim(0,5000)
    plt.ylim(-140,-40)
    plt.legend()
    if show:
        plt.show()
