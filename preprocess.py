# predictad/preprocess.py
"""
Pré-traitement: extraction de frames, transcript (si audio), features audio simples.
Dépend de moviepy et librosa.
"""
import os
from moviepy import VideoFileClip
import librosa
import soundfile as sf
import numpy as np
from PIL import Image
from tqdm import tqdm

def extract_audio_from_video(video_path, out_audio_path="extracted_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio_path, codec='pcm_s16le', logger=None)
    return out_audio_path

def extract_frames(video_path, out_dir="frames", fps=1):
    os.makedirs(out_dir, exist_ok=True)
    clip = VideoFileClip(video_path)
    duration = clip.duration  # durée exacte (float)
    print(f"Extraction des frames de la vidéo ({duration:.2f} secondes)...")

    saved = []
    # Générer les instants (en secondes) où extraire les frames
    times = np.arange(0, duration, 1/fps)

    for t in tqdm(times, desc="Extraction des frames"):
        try:
            frame = clip.get_frame(t)
            img = Image.fromarray(frame)
            p = os.path.join(out_dir, f"frame_{int(t*1000):06d}.jpg")  # millisecondes pour nommage
            img.save(p)
            saved.append(p)
        except Exception as e:
            continue

    clip.close()
    return saved


def extract_audio_features(audio_path, sr=22050):
    y, sr = librosa.load(audio_path, sr=sr)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    rms = librosa.feature.rms(y=y).mean()
    return {"tempo": float(tempo), "spectral_centroid": float(cent), "rms": float(rms)}

def save_transcript_stub(script_text, out_path="transcript.txt"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(script_text)
    return out_path
