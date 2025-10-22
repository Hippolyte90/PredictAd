# predictad/agents/linguistic_agent.py
from predictad import utils
from predictad.models_open_source import text_emotion_pipeline
import re

def analyze_script(script_text, config=None):
    words = re.findall(r"\w+", script_text)
    n_words = len(words)
    avg_word_len = sum(len(w) for w in words)/n_words if n_words>0 else 0
    ctas = ["subscribe","s'abonner","achetez","call to action","visitez","buy","shop","découvre", "abonnez"]
    has_cta = any(cta.lower() in script_text.lower() for cta in ctas)
    clarity = max(0, min(1, 1 - (n_words - 50)/300)) if n_words>0 else 0.5

    emotion_scores = text_emotion_pipeline(script_text)
    emotion_proxy = 0.0
    if emotion_scores:
        # si modèle renvoie émotions, on calcule un score agrégé (ex: sum of positive emotion scores)
        # adapte selon labels du modèle
        if "joy" in emotion_scores:
            emotion_proxy = emotion_scores.get("joy",0)
        else:
            # fallback: prendre max score
            emotion_proxy = max(emotion_scores.values()) if emotion_scores else 0.0

    # fallback heuristique si emotion_proxy trop faible ou none
    if emotion_proxy is None:
        emotion_proxy = 0.0

    return {
        "n_words": n_words,
        "avg_word_len": avg_word_len,
        "has_cta": has_cta,
        "clarity": float(clarity),
        "emotion_proxy": float(emotion_proxy)
    }
