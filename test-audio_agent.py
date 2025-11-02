# test_audio_analysis.py
from agents.audio_agent import analyze_audio
import pprint

# Chemin vers le fichier audio à analyser
audio_path = "output_audio.wav"  # Remplace par le chemin réel de ton fichier

# Appel de la fonction d'analyse
result = analyze_audio(audio_path, device="cpu")  # ou "cuda" si GPU disponible

# Affichage des résultats de manière lisible
print("\n=== Analyse Audio ===")
pprint.pprint(result)

# Affichage détaillé des MFCCs
if "mfccs" in result:
    print("\n=== MFCCs ===")
    for k, v in result["mfccs"].items():
        print(f"{k}: {v:.2f}")
