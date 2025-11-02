# predictad/agents/audio_agent.py
import librosa
import numpy as np
import os

def extract_audio_features(audio_path, sr=22050):
    """
    Extrait un ensemble complet de caractéristiques audio à partir d'un fichier.
    Paramètres :
        audio_path (str): chemin du fichier audio
        sr (int): fréquence d'échantillonnage
    Retourne :
        dict : dictionnaire des caractéristiques extraites
    """
    # Chargement du fichier audio
    y, sr = librosa.load(audio_path, sr=sr)
    
    # --- Caractéristiques temporelles ---
    rms = np.mean(librosa.feature.rms(y=y))
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))
    
    # --- Caractéristiques fréquentielles ---
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr))
    
    # --- Caractéristiques harmoniques ---
    y_harm, y_perc = librosa.effects.hpss(y)
    harmonic_ratio = np.mean(np.abs(y_harm)) / (np.mean(np.abs(y_perc)) + 1e-6)
    pitch, mag = librosa.piptrack(y=y, sr=sr)
    pitch_mean = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0.0
    
    # --- Caractéristiques rythmiques ---
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strength = np.mean(onset_env)
    
    # --- Caractéristiques perceptuelles ---
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = [float(np.mean(c)) for c in mfcc]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)
    
    # --- Caractéristiques dérivées ---
    spectral_flatness = np.mean(librosa.feature.spectral_flatness(y=y))
    
    # --- Compilation dans un dictionnaire ---
    features = {
        "rms": float(rms),
        "zcr": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "spectral_bandwidth": float(spectral_bandwidth),
        "spectral_rolloff": float(spectral_rolloff),
        "spectral_contrast": float(spectral_contrast),
        "spectral_flatness": float(spectral_flatness),
        "harmonic_ratio": float(harmonic_ratio),
        "pitch_mean": float(pitch_mean),
        "tempo": float(tempo),
        "beat_strength": float(beat_strength),
        "chroma_mean": float(chroma_mean),
    }
    
    # Ajout des MFCCs (coefficients cepstraux)
    for i, val in enumerate(mfcc_means):
        features[f"mfcc_{i+1}"] = val
    
    return features

def analyze_audio(audio_path, device=None):
    """
    Analyse un fichier audio et retourne les principales caractéristiques acoustiques,
    et un score d'énergie émotionnelle.
    """
    # Extraction des caractéristiques audio
    feats = extract_audio_features(audio_path)

    # Variables clés
    energy = feats.get("rms", 0)
    tempo = feats.get("tempo", 0)
    spectral_centroid = feats.get("spectral_centroid", 0)
    spectral_contrast = feats.get("spectral_contrast", 0)
    zcr = feats.get("zcr", 0)
    pitch_mean = feats.get("pitch_mean", 0)

    # Calcul d’un score d’énergie émotionnelle amélioré
    # (pondération empirique de différentes composantes sonores)
    emotion_energy = (
        0.4 * min(1.0, energy * 10) +
        0.2 * min(1.0, tempo / 200) +
        0.2 * min(1.0, spectral_centroid / 4000) +
        0.1 * min(1.0, spectral_contrast / 50) +
        0.1 * min(1.0, zcr * 10)
    )

    # Score final borné entre 0 et 1
    emotion_energy = float(min(1.0, emotion_energy))

    # Construction du dictionnaire de sortie
    result = {
        "tempo": float(tempo),
        "rms": float(energy),
        "zcr": float(zcr),
        "spectral_centroid": float(spectral_centroid),
        "spectral_contrast": float(spectral_contrast),
        "pitch_mean": float(pitch_mean),
        "emotion_energy": float(emotion_energy),
        "mfccs": {k: v for k, v in feats.items() if k.startswith("mfcc_")}
    }

    return result