from models_open_source import text_emotion_pipeline
from cta_model import cta_detect
import re

def analyze_script(script_text, config=None):
    words = re.findall(r"\w+", script_text)
    n_words = len(words)
    avg_word_len = sum(len(w) for w in words)/n_words if n_words>0 else 0
    has_cta = cta_detect(script_text)
    clarity = max(0, min(1, 1 - (n_words - 50)/300)) if n_words>0 else 0.5

    emotion_scores = text_emotion_pipeline(script_text)
    emotion_proxy = 0.0
    if emotion_scores:
        # si modèle renvoie émotions, on calcule un score agrégé (ex: sum of positive emotion scores)
        # adapte selon labels du modèle
        positive_labels = ["joy", "happiness", "surprise", "love", "excited", "grateful"]
        negative_labels = ["sadness", "anger", "fear", "disgust", "boredom", "neutral"]
        pos_score = sum(emotion_scores.get(label, 0.0) for label in positive_labels)
        neg_score = sum(emotion_scores.get(label, 0.0) for label in negative_labels)
        emotion_proxy = pos_score - neg_score  
        anger = emotion_scores.get("anger", 0.0)
        disgust = emotion_scores.get("disgust", 0.0)
        fear = emotion_scores.get("fear", 0.0)  
        joy = emotion_scores.get("joy", 0.0)
        neutral = emotion_scores.get("neutral", 0.0)
        sadness = emotion_scores.get("sadness", 0.0)
        surprise = emotion_scores.get("surprise", 0.0)
        love = emotion_scores.get("love", 0.0)
        excited = emotion_scores.get("excited", 0.0)
        grateful = emotion_scores.get("grateful", 0.0)
        
             
    # fallback heuristique si emotion_proxy trop faible ou none
    if emotion_proxy is None:
        emotion_proxy = 0.0

    return {
        "n_words": n_words,
        "avg_word_len": avg_word_len,
        "has_cta": has_cta,
        "clarity": float(clarity),
        "anger": float(anger),
        "disgust": float(disgust), 
        "fear": float(fear),
        "joy": float(joy),
        "neutral": float(neutral),
        "sadness": float(sadness),
        "surprise": float(surprise),
        "love": float(love),
        "excited": float(excited),
        "grateful": float(grateful),
        "emotion_proxy": float(emotion_proxy) 
    }
