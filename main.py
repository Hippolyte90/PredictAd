from preprocess import extract_audio_from_video, extract_images, audio_transcript
from agents.linguistic_agent import analyze_script
from agents.visual_agent import aggregate_images_analyses
from agents.audio_agent import analyze_audio
from agents.synth_agent import compute_scores, generate_report, plot_radar
from recommandation import generate_recommendations

import pprint

def treat_video_ad(video_path, out_audio_path= "output_audio.wav"):
    # Extraction de l’audio depuis la vidéo
    print("Extraction de l’audio depuis la vidéo...\n\n")
    audio_path = extract_audio_from_video(video_path, out_audio_path=out_audio_path)
    print(f"Audio extrait : {audio_path}")

    # Extraction de quelques frames (1 image par seconde ici)
    print("Extraction des frames depuis la vidéo...\n\n")
    frames = extract_images(video_path, out_dir="frames_extraites", fps=1)
    n_frames = len(frames)
    print(f"{n_frames} frames extraites. Exemple : {frames}")

    # Transciption de l’audio (voix → texte)
    print("Transcription de l’audio...\n\n")
    script_text = audio_transcript(audio_path, device="cpu")
    print(f"Transcription de l’audio :\n{script_text}")

    # Agent Linguistic
    print("Analyse du script...\n\n")
    text_features = analyze_script(script_text)

    # Agent visuel
    print("Analyse des images extraites...\n\n")
    visual_features = aggregate_images_analyses(frames, script_text=script_text)

    # Agent Audio
    print("Analyse de l’audio extrait...\n\n")
    audio_features = analyze_audio(audio_path)

    # Agent Synthèse
    print("Calcul des scores video...\n\n")
    video_scores = compute_scores(audio_features, visual_features, text_features)
    
    return video_scores


    
# Pour tester la fonction treat_video_ad

# Chemin vers la vidéo à traiter
#video_path = "video-test-en.mp4"

# Traiter la vidéo publicitaire
#video_scores = treat_video_ad(video_path)
#print(video_scores)
#generate_report(video_scores)
#recomendations = generate_recommendations(video_scores)

#pprint(recomendations)
#plot_radar(video_scores)

