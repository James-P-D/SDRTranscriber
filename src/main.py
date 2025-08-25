from rtlsdr import RtlSdr
from scipy import signal
from time import sleep
import numpy as np
from scipy.io import wavfile
import threading
import queue
from pathlib import Path
from pydub import AudioSegment, silence
import whisper # pip install openai-whisper

#todo: remove me
import warnings
warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

sdr = RtlSdr()
samples = queue.Queue()
sounds = queue.Queue()


class ReadThread(threading.Thread):
    def __init__(self, sdr, samples):
        threading.Thread.__init__(self)
        self.srd = sdr
        self.samples = samples
        self.stopit = False

    def run(self):
        while not self.stopit:
            self.samples.put(sdr.read_samples(8192000 / 4))


class ProcessThread(threading.Thread):
    def __init__(self, samples, sounds):
        threading.Thread.__init__(self)
        self.samples = samples
        self.sounds = sounds
        self.stopit = False

    def run(self):
        while not self.stopit:
            samples = self.samples.get()

            x1 = np.array(samples).astype("complex64")
            fc1 = np.exp(-1.0j * 2.0 * np.pi * F_offset / Fs * np.arange(len(x1)))
            x2 = x1 * fc1
            bandwidth = 2500  # khz broadcast radio.
            n_taps = 64
            # Use Remez algorithm to design filter coefficients
            lpf = signal.remez(n_taps, [0, bandwidth, bandwidth + (Fs / 2 - bandwidth) / 4, Fs / 2], [1, 0], fs=Fs)
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
            x7 *= 10000 / np.max(np.abs(x7))
            self.sounds.put(x7)

class WriteThread(threading.Thread):
    def __init__(self, thread2, sounds):
        threading.Thread.__init__(self)
        self.thread2 = thread2
        self.sounds = sounds
        self.stopit = False
        self.counter = 0

    def run(self):
        while not self.stopit:
            x7 = self.sounds.get()
            wavfile.write(f'{self.counter:08d}.wav', int(self.thread2.Fs_audio), x7.astype("int16"))
            self.counter += 1


folder = Path(".")
# Delete all .wav files
for file in folder.glob("*.wav"):
    file.unlink()

thread1 = ReadThread(sdr, samples)
thread2 = ProcessThread(samples, sounds)
thread1.start()
thread2.start()

# configure device
#Freq = 440.713e6  # mhz
#Freq = 107600000
Freq =  93500000
Fs = 1140000
F_offset = 2500
Fc = Freq - F_offset
sdr.sample_rate = Fs
sdr.center_freq = Fc
sdr.gain = 'auto'
Fs_audio = 0
counter = 0

thread3 = WriteThread(thread2, sounds)
thread3.start()


def delete_file(filename):
    folder = Path(".")
    filepath = folder / filename
    if filepath.exists():
        filepath.unlink()
        #print(f"Deleted {filename}")


def delete_files(start, end):
    folder = Path(".")
    for i in range(start, end+1):
        filename = f"{i:08d}.wav"  # zero-padded 8 digits
        filepath = folder / filename
        if filepath.exists():
            filepath.unlink()


try:
    # Initialise offline speech recognition
    model = whisper.load_model("base") # options: tiny, base, small, medium, large

    while True:
        sleep(20)

        files = sorted([f.name for f in Path(".").glob("????????.wav")])
        if len(files) < 3:
            #print("not enough files to process yet")
            continue
        first_file = files[0].replace(".wav", "")
        last_file = files[-2].replace(".wav", "")
        true_last_file = ""

        combined = AudioSegment.from_wav(files[0])

        duration_ms = len(combined)
        duration_sec = duration_ms / 1000

        for f in files[1:-1]:
            segment = AudioSegment.from_wav(f)
            combined += segment

        combined_file = f"{first_file}_{last_file}.wav"
        combined.export(combined_file, format="wav")
        silence_ranges = silence.detect_silence(
            combined,
            min_silence_len=500,  # minimum length of silence (ms)
            silence_thresh=combined.dBFS - 16  # silence threshold (dB)
        )

        if not silence_ranges:
            # If no silent gaps in our combined file, then delete it. Next time hopefully we'll have more
            # .wav files to create a bigger combined_file which *does* contain some silence.
            delete_file(combined_file)
            continue

        start_sample = 0
        for end_sample in [a for a, _ in silence_ranges]:
            chunk = combined[start_sample:end_sample]
            chunk_filename = f"{first_file}_{last_file}__{start_sample:020d}_{end_sample:020d}.wav"
            chunk.export(chunk_filename, format="wav")
            true_last_file = files[int(end_sample/duration_ms)].replace(".wav", "")

            try:
                result = model.transcribe(chunk_filename)
                print("Transcription:", result["text"])
            except Exception as e:
                print("Speech recognition error:", e)
            delete_file(chunk_filename)
            start_sample = end_sample

        delete_files(int(first_file), int(true_last_file))
        delete_file(combined_file)

except KeyboardInterrupt:
    print("bye")
    thread1.stopit = True
    thread2.stopit = True
    thread3.stopit = True

