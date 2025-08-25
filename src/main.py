import matplotlib.pyplot as plt
from rtlsdr import RtlSdr
from scipy import signal
from time import sleep
import numpy as np
from scipy.io import wavfile
import sounddevice
import threading
import queue
from pathlib import Path
import time
from pydub import AudioSegment, silence
import whisper # pip install openai-whisper

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
        #print("Starting read")
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
            #print('getting')
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
            #print("processed")

#########################################################
class writeThread(threading.Thread):
    def __init__(self, thread2, sounds):
        threading.Thread.__init__(self)
        self.thread2 = thread2
        self.sounds = sounds
        self.stopit = False
        self.counter = 0

    def run(self):
        while not self.stopit:
            x7 = self.sounds.get()
            # sounddevice.play(x7,Fs_audio)
            # x7.astype("int16").tofile("wbfm-mono.raw")  #Raw file.
            # need to look at this: https://stackoverflow.com/questions/54936484/updating-appending-to-a-wav-file-in-python
            wavfile.write(f'{self.counter:08d}.wav', int(self.thread2.Fs_audio), x7.astype("int16"))
            self.counter += 1


# Path to directory
folder = Path(".")

# Delete all .wav files
for file in folder.glob("*.wav"):
    #print(f"Deleting {file.name}")
    file.unlink()

thread1 = readThread(sdr, samples)
thread2 = processThread(samples, sounds)
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

thread3 = writeThread(thread2, sounds)
thread3.start()


def delete_file(filename):
    folder = Path(".")
    filepath = folder / filename
    if filepath.exists():
        filepath.unlink()
        #print(f"Deleted {filename}")


def delete_files(start, end):
    folder = Path(".")
    for i in range(start, end):
        filename = f"{i:08d}.wav"  # zero-padded 8 digits
        filepath = folder / filename
        if filepath.exists():
            filepath.unlink()
            #print(f"Deleted {filename}")
        else:
            #print(f"Cant find {filename}")
            pass



try:
    # Initialise offline speech recognition
    model = whisper.load_model("base") # options: tiny, base, small, medium, large

    while True:
        sleep(20)

        files = sorted([f.name for f in Path(".").glob("????????.wav")])
        #print(f"files = {files}")
        if len(files) < 3:
            #print("not enough files to process yet")
            continue
        first_file = files[0].replace(".wav", "")
        last_file = files[-2].replace(".wav", "")
        true_last_file = ""

        #print(f"processing {files[0]}")
        combined = AudioSegment.from_wav(files[0])

        duration_ms = len(combined)
        duration_sec = duration_ms / 1000
        #print(f"Duration: {duration_ms}\t{duration_sec}")

        for f in files[1:-1]:
            #print(f"processing {f}")
            segment = AudioSegment.from_wav(f)
            #duration_ms = len(segment)
            #duration_sec = duration_ms / 1000
            #print(f"Duration: {duration_ms}\t{duration_sec}")
            combined += segment

        combined_file = f"{first_file}_{last_file}.wav"
        combined.export(combined_file, format="wav")
        #print(f"written combined file: {combined_file}")
        silence_ranges = silence.detect_silence(
            combined,
            min_silence_len=1000,  # minimum length of silence (ms)
            silence_thresh=combined.dBFS - 16  # silence threshold (dB)
        )
        if not silence_ranges:
            print("No silent points found")
            delete_file(combined_file)
            continue

        print(f"{first_file}")
        print(f"{last_file}")

        #print("Silent parts (ms):", silence_ranges)

        #midpoints_silences = [(a + b) / 2 for a, b in silence_ranges]
        #print("Silent parts (ms):", midpoints_silences)

        start_sample = 0
        for end_sample in [a for a, _ in silence_ranges]: #midpoints_silences[:-1]:
            chunk = combined[start_sample:end_sample]
            #print(f"chunk from {start_sample} to {end_sample}")
            chunk_filename = f"{first_file}_{last_file}__{start_sample:020d}_{end_sample:020d}.wav"
            chunk.export(chunk_filename, format="wav")
            true_last_file = files[int(end_sample/duration_ms)].replace(".wav", "")
            print(f"true_last_file updated to {true_last_file}")
            #with sr.AudioFile(chunk_filename) as source:
            #    audio = r.record(source)  # read the whole file

            #print(chunk_filename)
            # Recognize (using Google Web Speech API by default)
            #try:
            #    text = r.recognize_google(audio)
            #    print("ONLINE : Transcription:", text)
            #except sr.UnknownValueError:
            #    print("ONLINE : Speech not understood")
            #except sr.RequestError as e:
            #    print("ONLINE : API unavailable:", e)

            try:
                result = model.transcribe(chunk_filename)
                #if len(result["text"]) > 0:
                print("Transcription:", result["text"])
            except Exception as e:
                print("Speech recognition error:", e)
            #print("")
            #delete_file(chunk_filename)
            start_sample = end_sample

        print(f"About to delete range of files from {first_file} to {true_last_file}")
        delete_files(int(first_file), int(true_last_file))
        delete_file(combined_file)
        #print("Done")
        #print("-----------------------------------------------------------------")





except KeyboardInterrupt:
    print("bye")
    thread1.stopit = True
    thread2.stopit = True
    thread3.stopit = True