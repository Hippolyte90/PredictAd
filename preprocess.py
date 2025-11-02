# predictad/preprocess.py
"""
Pré-traitement: extraction de frames, transcript (si audio), features audio simples.
Dépend de moviepy et librosa.
"""
import os
from moviepy.editor import VideoFileClip # from moviepy.edictor import VideoFileClip for local runing beacause of moviepy version of python 12
import soundfile as sf
import numpy as np
from PIL import Image
from tqdm import tqdm
from models_open_source import transcribe_audio_whisper
from message_gpt import config_env

config_env()


def extract_audio_from_video(video_path, out_audio_path="extracted_audio.wav"):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(out_audio_path, codec='pcm_s16le', logger=None)
    return out_audio_path

def extract_images(video_path, out_dir="frames", fps=1):
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


def audio_transcript(audio_path, device=None):
    # Transcription du fichier audio (voix → texte)
    try:
        transcription = transcribe_audio_whisper(audio_path, device=device)
    except Exception:
        transcription = None
    return transcription