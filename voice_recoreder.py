import sounddevice as sd
import numpy as np
import soundfile as sf
import noisereduce as nr
from scipy.signal import butter, filtfilt
import time
import sys

SAMPLE_RATE = 16000
DURATION = 7                 # total recording duration (sec)
NOISE_DURATION = 0.5         # first 0.5s used as noise
COUNTDOWN = 3                # seconds before recording
OUTPUT_FILE = "voice_clean.wav"

# -----------------------------
# Helper: High-pass filter
# -----------------------------
def highpass_filter(data, cutoff=80, fs=16000, order=4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="high", analog=False)
    return filtfilt(b, a, data)


print("🎙️ Recording will start in:")
for i in range(COUNTDOWN, 0, -1):
    sys.stdout.write(f"\r{i} ")
    sys.stdout.flush()
    time.sleep(1)
print("\rGo! Stay quiet for first 0.5s ")


audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype="float32",
)
sd.wait()
audio = audio.flatten()

noise_samples = int(NOISE_DURATION * SAMPLE_RATE)
noise_clip = audio[:noise_samples]

# -----------------------------
# Noise reduction
# -----------------------------
reduced_noise = nr.reduce_noise(
    y=audio,
    y_noise=noise_clip,
    sr=SAMPLE_RATE,
    prop_decrease=1.0,
)

# -----------------------------
# High-pass filter
# -----------------------------
clean_audio = highpass_filter(reduced_noise, cutoff=80, fs=SAMPLE_RATE)
clean_audio /= np.max(np.abs(clean_audio) + 1e-9)

# -----------------------------
# Save output
# -----------------------------
sf.write(OUTPUT_FILE, clean_audio, SAMPLE_RATE)
print(f"✅ Clean voice saved to {OUTPUT_FILE}")
