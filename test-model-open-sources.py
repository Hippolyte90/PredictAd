# test_models_open_source.py
import os
from models_open_source import (
    transcribe_audio_whisper,
    image_caption_blip,
    clip_image_text_similarity,
    text_emotion_pipeline, download_youtube_video
)

def test_text_emotion():
    print("\n=== Test: Text Emotion Detection ===")
    text = """Bonjour à tous! Aujourd'hui, nous allons découvrir les merveilles de la nature. N'oubliez pas de vous abonner à notre chaîne pour plus de vidéos incroyables! Visitez notre site web pour en savoir plus."""
    result = text_emotion_pipeline(text)
    if result is None:
        print("❌ Text emotion model not available.")
    else:
        print("✅ Emotion scores:")
        for label, score in result.items():
            print(f"  {label}: {score:.3f}")

def test_image_caption():
    print("\n=== Test: Image Caption (BLIP) ===")
    image_path = "frames_extraites/frame_012000.jpg"
    if not os.path.exists(image_path):
        print("⚠️ Aucune image trouvée. Télécharge ou place un fichier 'sample.jpg'.")
        return
    caption = image_caption_blip(image_path)
    if caption is None:
        print("❌ BLIP non disponible.")
    else:
        print(f"✅ Caption générée : {caption}")

def test_clip_similarity():
    print("\n=== Test: CLIP Image-Text Similarity ===")
    image_path = "frames_extraites/frame_012000.jpg"
    if not os.path.exists(image_path):
        print("⚠️ Place un fichier 'sample.jpg' dans le dossier avant de tester.")
        return
    text = "A beautiful landscape with mountains."
    score = clip_image_text_similarity(image_path, text)
    if score is None:
        print("❌ CLIP non disponible.")
    else:
        print(f"✅ Similarité CLIP (texte-image) : {score:.3f}")

def test_whisper_transcription():
    print("\n=== Test: Whisper Audio Transcription ===")
    audio_path = "output_audio.wav"
    if not os.path.exists(audio_path):
        print("⚠️ Place un fichier audio 'sample.wav' (petit extrait) dans le dossier.")
        return
    text = transcribe_audio_whisper(audio_path)
    if text is None:
        print("❌ Whisper non disponible.")
    else:
        print(f"✅ Transcription : {text}")



if __name__ == "__main__":
    #test_text_emotion()
    #test_image_caption()
    #test_clip_similarity()
    #test_whisper_transcription()
    video_url = "https://youtu.be/PEZo9mxbqo8"
    path = download_youtube_video(video_url)
    print(f"✅ Video downloaded successfully!\n📁 File path: {path}")
