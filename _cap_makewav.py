import wave
import array
import math
import random

RATE = 16000
random.seed(7)
DUR = 2.5

samples = array.array("h")
freq = 220.0
for i in range(int(RATE * DUR)):
    wobble = 6.0 * math.sin(2 * math.pi * 3.0 * i / RATE)
    value = 9000 * math.sin(2 * math.pi * (freq + wobble) * i / RATE)
    value += random.gauss(0, 1500)
    samples.append(max(-32767, min(32767, int(value))))

with wave.open("_cap_audio_speechlike.wav", "w") as handle:
    handle.setnchannels(1)
    handle.setsampwidth(2)
    handle.setframerate(RATE)
    handle.writeframes(samples.tobytes())
print("wav-speechlike-written")
