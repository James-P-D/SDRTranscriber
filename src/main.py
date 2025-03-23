import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy import signal
from time import sleep
import numpy as np
from scipy.io import wavfile
import sounddevice
import threading
import queue

sdr = RtlSdr()
samples = queue.Queue()
sounds = queue.Queue()


class readThread(threading.Thread):
    def __init__(self, sdr, samples):
        threading.Thread.__init__(self)
        self.srd = sdr
        self.samples = samples
        self.stopit = False

    def run(self):
        print("Starting read")
        while not self.stopit:
            self.samples.put(sdr.read_samples(8192000 / 4))


class processThread(threading.Thread):
    def __init__(self, samples, sounds):
        threading.Thread.__init__(self)
        self.samples = samples
        self.sounds = sounds
        self.stopit = False

    def run(self):
        while not self.stopit:
            print('getting')
            samples = self.samples.get()

            x1 = np.array(samples).astype("complex64")
            fc1 = np.exp(-1.0j * 2.0 * np.pi * F_offset / Fs * np.arange(len(x1)))
            x2 = x1 * fc1
            bandwidth = 2500  # khz broadcast radio.
            n_taps = 64
            # Use Remez algorithm to design filter coefficients
            lpf = signal.remez(n_taps, [0, bandwidth, bandwidth + (Fs / 2 - bandwidth) / 4, Fs / 2], [1, 0], Hz=Fs)
            x3 = signal.lfilter(lpf, 1.0, x2)
            dec_rate = int(Fs / bandwidth)
            x4 = x3[0::dec_rate]
            Fs_y = Fs / dec_rate
            f_bw = 200000
            dec_rate = int(Fs / f_bw)
            x4 = signal.decimate(x2, dec_rate)
            # Calculate the new sampling rate
            Fs_y = Fs / dec_rate
            y5 = x4[1:] * np.conj(x4[:-1])
            x5 = np.angle(y5)

            # The de-emphasis filter
            # Given a signal 'x5' (in a numpy array) with sampling rate Fs_y
            d = Fs_y * 75e-6  # Calculate the # of samples to hit the -3dB point
            x = np.exp(-1 / d)  # Calculate the decay between each sample
            b = [1 - x]  # Create the filter coefficients
            a = [1, -x]
            x6 = signal.lfilter(b, a, x5)
            audio_freq = 41000.0
            dec_audio = int(Fs_y / audio_freq)
            Fs_audio = Fs_y / dec_audio
            self.Fs_audio = Fs_audio
            x7 = signal.decimate(x6, dec_audio)

            # sounddevice.play(x7,Fs_audio)
            x7 *= 10000 / np.max(np.abs(x7))
            self.sounds.put(x7)
            print("processed")


thread1 = readThread(sdr, samples)
thread2 = processThread(samples, sounds)
thread1.start()
thread2.start()

# configure device
Freq = 107600000# 440.713e6  # mhz
Fs = 1140000
F_offset = 2500
Fc = Freq - F_offset
sdr.sample_rate = Fs
sdr.center_freq = Fc
sdr.gain = 'auto'
Fs_audio = 0
counter = 0
try:
    while True:
        x7 = sounds.get()
        # sounddevice.play(x7,Fs_audio)
        # x7.astype("int16").tofile("wbfm-mono.raw")  #Raw file.
        # need to look at this: https://stackoverflow.com/questions/54936484/updating-appending-to-a-wav-file-in-python
        wavfile.write(f'wav{counter:03}.wav',int(thread2.Fs_audio), x7.astype("int16"))

        print('playing...')
        sounddevice.play(x7.astype("int16"), thread2.Fs_audio, blocking=True)
        counter += 1
except KeyboardInterrupt:
    print("bye")
    thread1.stopit = True
    thread2.stopit = True