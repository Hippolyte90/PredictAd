import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# Charger l'audio
audio_path = "output_audio.wav"
y, sr = librosa.load(audio_path)

# Calcul des caractéristiques
tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
tempo = float(tempo)
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
rms = librosa.feature.rms(y=y)

# Création d'une figure avec 3 sous-graphes
plt.figure(figsize=(14, 8))

# 1️⃣ Forme d'onde
plt.subplot(3, 1, 1)
librosa.display.waveshow(y, sr=sr)
plt.title("Forme d'onde de l'audio")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude")

# 2️⃣ RMS (puissance moyenne)
plt.subplot(3, 1, 2)
frames = range(len(rms[0]))
t = librosa.frames_to_time(frames, sr=sr)
plt.plot(t, rms[0], color='r')
plt.title(f"RMS du signal (Puissance moyenne) - Tempo estimé : {tempo:.1f} BPM")
plt.xlabel("Temps (s)")
plt.ylabel("Amplitude RMS")

# 3️⃣ Spectral Centroid
plt.subplot(3, 1, 3)
plt.semilogy(t, spectral_centroid[0], color='g')
plt.title("Spectral Centroid (Centre de gravité fréquentiel)")
plt.xlabel("Temps (s)")
plt.ylabel("Fréquence (Hz)")

plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 4))
S = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(S))
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='log')
plt.colorbar(format="%+2.0f dB")
plt.title("Spectrogramme")
plt.show()
