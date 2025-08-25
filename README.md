# SDRTranscriber
SDR audio transcriber in Python

![Example](https://github.com/James-P-D/SDRTranscriber/blob/main/screenshot.jpg)

## Introduction

I was interested in keeping track of a certain frequency on my SDR, and whilst lots of software exists to record audio, the channel is rarely in use so would have to trawl through 100s of hours of audio trying to find the actual places when people are speaking.

This script which continuously records 1-2 second snippets and saves them as files named `00000000.wav`, `00000001.wav` and so on. After every 20 seconds we concatenate the files into a larger one named `00000000_00000035.wav` and analyse the file for periods of silence lasting 500ms or longer. We then snip the file back into smaller chunks which be believe contain full sentences. These files are saved as `00000000_00000035_x_y.wav` where `x` and `y` represent the start and end period of the audio in ms. We then pass this .wav file into the offline OpenAI-Whisper speech recognition library. Finally we clear up any old .wav files that have been processed and repeat the process.

## Setup

The project was tested with a cheap RTL-SDR device ([details on where to buy can be found here](https://www.rtl-sdr.com/buy-rtl-sdr-dvb-t-dongles/)):

![RTLSDR](https://github.com/James-P-D/SDRTranscriber/blob/main/rtlsdr.jpg)

The project requires a number of libraries:

```
pip install argparse
pip install rtlsdr
pip install scipy
pip install openai-whisper
```

## Usage

Running the program without arguments will produce the following:

```
C:\Users\jdorr\OneDrive\Desktop\Dev\SDRTranscriber\src\>python.exe main.py
usage: main.py [-h] -f [Frequency]
main.py: error: the following arguments are required: -f
```

Simply use the `-f` argument to specify the frequency. In the example screenshot above I am listening to [BBC Radio 4](https://en.wikipedia.org/wiki/BBC_Radio_4) so I'm supplying the argument `-f 93500000` for 93.5MHz.

Example output:

```
C:\Users\jdorr\OneDrive\Desktop\Dev\SDRTranscriber\src\>python.exe main.py -f 93500000
Found Rafael Micro R820T/2 tuner
Exact sample rate is: 1140000.002265 Hz
Transcription:  And so a woman who comes into the married couple's lives, hired often by the wronged wife.
Transcription:  her way into the lives of the husband and the mistress and breaks up the affair. It's a
really fascinating idea made jaw-dropping by a new documentary which has filmed the whole experience from
all points of view. I spoke to the director Elizabeth Lowe earlier who told me more about the mistress
disbellars and their work.
Transcription:  If this new emerging industry in China that's only
```