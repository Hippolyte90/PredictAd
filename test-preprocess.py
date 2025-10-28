from preprocess import extract_audio_from_video, extract_frames, extract_audio_features, save_transcript_stub

# 1️⃣ Chemin vers la vidéo à traiter
video_path = "video-test.mp4"

# 2️⃣ Extraction de l’audio depuis la vidéo
audio_path = extract_audio_from_video(video_path, out_audio_path="output_audio.wav")
print(f"Audio extrait : {audio_path}")

# 3️⃣ Extraction de quelques frames (1 image par seconde ici)
frames = extract_frames(video_path, out_dir="frames_extraites", fps=1)
print(f"{len(frames)} frames extraites. Exemple : {frames[:3]}")

# 4️⃣ Extraction de caractéristiques audio simples
features = extract_audio_features(audio_path)
print("Caractéristiques audio :")
print(features)
# Exemple de sortie :
# {'tempo': 120.3, 'spectral_centroid': 2034.5, 'rms': 0.045}

# 5️⃣ Création d’un faux transcript (pour test)
texte = "Ceci est un exemple de transcript généré pour la démonstration."
transcript_path = save_transcript_stub(texte, out_path="transcript.txt")
print(f"Transcript sauvegardé dans : {transcript_path}")
