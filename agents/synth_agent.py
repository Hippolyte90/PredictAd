# predictad/agents/synth_agent.py
import numpy as np
import matplotlib.pyplot as plt

def normalize(value, vmin, vmax):
    """Normalise une valeur entre 0 et 1."""
    return max(0, min(1, (value - vmin) / (vmax - vmin)))

def compute_scores(audio, image, script):
    """Calcule les scores principaux à partir des sorties brutes des 3 agents."""

    # === 1. SCORE AUDIO ===
    tempo_norm = normalize(audio.get("tempo", 0), 60, 180) # Rythme de la vidéo
    rms_norm = normalize(audio.get("rms", 0), 0.02, 0.18) # Volume perçu
    emotion_energy = audio.get("emotion_energy", 0) # Intensité émotionnelle moyenne
    audio_score = np.mean([tempo_norm, rms_norm, emotion_energy])

    # === 2. SCORE VISUEL ===
    brightness_norm = normalize(image.get("mean_brightness", 0), 30, 200) # Luminosité modérée
    text_sim = image.get("mean_text_similarity", 0.2) # Cohérence texte–image
    visual_score = 0.3 * brightness_norm + 0.7 * text_sim

    # === 3. SCORE LINGUISTIQUE ===
    clarity = script.get("clarity", 0) # Clarté du message
    cta = 1.0 if script.get("has_cta", False) else 0.0 # Présence d'un CTA
    joy = script.get("joy", 0.0) # Score de joie
    emotion_proxy = script.get("emotion_proxy", 0.0) # Intensité émotionnelle globale
    linguistic_score = 0.3 * clarity + 0.3 * joy + 0.2 * cta + 0.2 * emotion_proxy

    # === 4. SCORE GLOBAL ===
    final_score = 0.4 * linguistic_score + 0.35 * audio_score + 0.25 * visual_score

    # === 5. Structurer les résultats ===
    scores = {
        "cta": script.get("has_cta", False),
        "Audio": round(audio_score * 100, 1),
        "Visuel": round(visual_score * 100, 1),
        "Linguistique": round(linguistic_score * 100, 1),
        "Global": round(final_score * 100, 1),
    }

    return scores

def generate_report(scores):
    """Affiche un rapport simple."""
    print("\n===== RAPPORT D'ANALYSE PREDICTAD =====")
    print(f"🎧 Score Audio       : {scores['Audio']} / 100")
    print(f"👁️  Score Visuel      : {scores['Visuel']} / 100")
    print(f"🧠 Score Linguistique : {scores['Linguistique']} / 100")
    print(f"🌍 Score Global       : {scores['Global']} / 100\n")

    # Interprétation simple
    if scores['Global'] < 40:
        niveau = "Pub à faible impact"
    elif scores['Global'] < 60:
        niveau = "Pub à Impact moyen"
    elif scores['Global'] < 80:
        niveau = "Pub à bon impact"
    else:
        niveau = "Pub à excellent impact"

    print(f"➡️ Interprétation : {niveau}")
    
    return niveau

def plot_radar(scores):
    """Trace un graphique radar des scores."""
    categories = list(scores.keys())[1:-1]  # sauf Global
    # print(categories)
    values = [scores[k] for k in categories]
    values += [values[0]]  # fermer le polygone

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += [angles[0]]

    fig = plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Profil d'impact publicitaire")
    plt.show()
    return fig
    
