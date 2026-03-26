# Audio Noise Reduction

## AIM

To perform **audio noise reduction and analysis** using Python by applying noise reduction techniques and visualizing the results in both time and frequency domains.

---
## TASK

- Upload and load an audio signal  
- Apply noise reduction and optional gain processing  
- Save and compare original and processed audio  
- Perform STFT to analyze frequency components  
- Plot spectrograms of original, processed, and noise signals  
- Visualize signals in time domain and compare results  

---

## APPARATUS REQUIRED

- PC with Python (Google Colab / Jupyter Notebook)  
- Libraries:
  - `librosa`
  - `noisereduce`
  - `pedalboard`
  - `soundfile`
  - `matplotlib`
  - `numpy`

---

## THEORY

Noise in audio signals reduces clarity and quality. Noise reduction techniques remove unwanted components while preserving useful information.

- **Noise Reduction:** Removes background noise  
- **STFT:** Converts signal to time-frequency domain  
- **Spectrogram:** Shows frequency variation over time  
- **dB Scale:** Helps visualize signal intensity  
- **Gain:** Adjusts signal amplitude  

---

## PROGRAM: 

~~~
# =========================
# INSTALL LIBRARIES
# =========================
!pip install noisereduce pedalboard librosa soundfile

# =========================
# IMPORT LIBRARIES
# =========================
import librosa
import librosa.display
import noisereduce as nr
from pedalboard import Pedalboard, Gain
from IPython.display import Audio, display
from google.colab import files
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np

# =========================
# UPLOAD AUDIO FILE
# =========================
print("Please upload your audio file (.wav recommended)")
uploaded = files.upload()

if uploaded:
    file_name = list(uploaded.keys())[0]
    file_path = file_name
    print(f"Loading audio from: {file_path}")
else:
    raise FileNotFoundError("No audio file uploaded.")

# =========================
# LOAD AUDIO
# =========================
y, sr = librosa.load(file_path, sr=None)

# =========================
# NOISE REDUCTION
# =========================
y_denoised = nr.reduce_noise(y=y, sr=sr, prop_decrease=1.0)

# =========================
# OPTIONAL EFFECT (GAIN)
# =========================
board = Pedalboard([Gain(gain_db=0)])  # Change gain if needed
y_processed = board(y_denoised, sr)

# =========================
# SAVE OUTPUT
# =========================
output_file_name = "output_denoised_audio.wav"
sf.write(output_file_name, y_processed, sr)

# =========================
# PLAY AUDIO
# =========================
print("Original Audio:")
display(Audio(file_path))

print("Denoised Audio:")
display(Audio(output_file_name))

# =========================
# SPECTROGRAM ANALYSIS
# =========================
D_original = librosa.stft(y)
D_processed = librosa.stft(y_processed)

DB_original = librosa.amplitude_to_db(np.abs(D_original), ref=np.max)
DB_processed = librosa.amplitude_to_db(np.abs(D_processed), ref=np.max)

plt.figure(figsize=(15, 8))

plt.subplot(2, 1, 1)
librosa.display.specshow(DB_original, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Original Audio Spectrogram')

plt.subplot(2, 1, 2)
librosa.display.specshow(DB_processed, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Denoised Audio Spectrogram')

plt.tight_layout()
plt.show()

# =========================
# NOISE SPECTROGRAM
# =========================
noise_signal = y - y_processed
D_noise = librosa.stft(noise_signal)
DB_noise = librosa.amplitude_to_db(np.abs(D_noise), ref=np.max)

plt.figure(figsize=(12, 6))
librosa.display.specshow(DB_noise, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Estimated Noise Spectrogram')
plt.tight_layout()
plt.show()

# =========================
# TIME DOMAIN PLOTS
# =========================
time = np.linspace(0, len(y) / sr, num=len(y))

plt.figure(figsize=(15, 5))

plt.subplot(2, 1, 1)
plt.plot(time, y, alpha=0.7)
plt.title('Original Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, y_processed, alpha=0.7)
plt.title('Denoised Signal')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid()

plt.tight_layout()
plt.show()

print("Noise reduction completed successfully!")
~~~

# OUTPUT: 
### Spectrogram of Original and Denoised Signals:

<img width="1000" height="400" alt="image" src="https://github.com/user-attachments/assets/b92462e7-bbba-4fa8-af35-3f58fb3ef869" />

### Plot of Original and Denoised Signals:

<img width="1000" height="400" alt="download" src="https://github.com/user-attachments/assets/b7c39acf-f17a-4839-8cba-5145afc3a800" />

---

## FUTURE IMPROVEMENTS

- Real-time noise reduction  
- Advanced filtering techniques  
- GUI-based implementation

---
  
# RESULT: 
  Thus Audio noise reduction was effectively implemented, and analysis confirms improved signal quality in both time and frequency domains.
