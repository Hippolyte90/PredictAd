# predictad/agents/audio_agent.py
from PredictAd.preprocess import extract_audio_features
from PredictAd.models_open_source import transcribe_audio_whisper
import numpy as np
import os

def analyze_audio(audio_path, device=None):
    feats = extract_audio_features(audio_path)
    transcription = None
    try:
        transcription = transcribe_audio_whisper(audio_path, device=device)
    except Exception:
        transcription = None

    energy = feats.get("rms", 0)
    tempo = feats.get("tempo", 0)
    emotion_energy = min(1.0, (energy*10 + tempo/200))

    return {
        "tempo": float(tempo),
        "rms": float(energy),
        "emotion_energy": float(emotion_energy),
        "transcription": transcription
    }
