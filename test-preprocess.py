from preprocess import extract_audio_from_video, extract_images

# 1️⃣ Chemin vers la vidéo à traiter
video_path = "video-test-en.mp4"

# 2️⃣ Extraction de l’audio depuis la vidéo
audio_path = extract_audio_from_video(video_path, out_audio_path="output_audio.wav")
print(f"Audio extrait : {audio_path}")

# 3️⃣ Extraction de quelques frames (1 image par seconde ici)
frames = extract_images(video_path, out_dir="frames_extraites", fps=1)
print(f"{len(frames)} frames extraites. Exemple : {frames}")
