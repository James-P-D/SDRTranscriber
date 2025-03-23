import numpy as np
import scipy.signal
import array
import rtlsdr
import pyaudio
import queue

import signal
def signal_handler(signum, frame):
    exit(-1)
signal.signal(signal.SIGINT, signal_handler)

Fs=3.2e6                       # changed from  1.2e6 # sampling rate
tune= 107600000                # changed from 82.5e6
gain = 30 # LNA gain
length=1024*50

sdr = rtlsdr.RtlSdr(0)
sdr.set_sample_rate(Fs)
sdr.set_manual_gain_enabled(1)
sdr.set_gain(gain)
sdr.set_center_freq(tune)

pa = pyaudio.PyAudio()

que = queue.Queue()

def callback(in_data, frame_count, time_info, status):
    capture = que.get()
    # decimate 1/5 from 1.2MHz to 240kHz
    sigif = scipy.signal.decimate(capture, 5, ftype='iir')
    # convert to continuous phase angle
    phase = np.unwrap(np.angle(sigif))
    # differentiate phase brings into frequency
    pd = np.convolve(phase, [1,-1], mode='valid')
    # decimate 1/10 from 240kHz to 24kHz
    audio = scipy.signal.decimate(pd, 10, ftype='iir')
    # make binary buffer from numpy array for pyaudio
    buf = array.array('f', audio).tobytes()             # Changed to tobytes() since tostring() deprecated
    return (buf, pyaudio.paContinue)

# audio rate is 1.2MHz/(5*10) = 24kHz
stream = pa.open(format=pyaudio.paFloat32,
                channels=1, rate=int(Fs/50), output=True, stream_callback = callback)
stream.start_stream()

def capture_callback(capture, rtlsdr_obj):
    print("capture_callback()")
    que.put(capture)

sdr.read_samples_async(capture_callback, length)

stream.stop_stream()
pa.close()
sdr.close()