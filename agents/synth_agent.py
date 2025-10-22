# predictad/agents/synth_agent.py
from predictad import utils
import json

def synthesize(linguistic_res, visual_res, audio_res, config):
    w = config.get("weights", {"emotion":0.4,"brand_coherence":0.3,"creativity":0.2,"clarity":0.1})
    emotion_score = (linguistic_res.get("emotion_proxy",0) + audio_res.get("emotion_energy",0) + visual_res.get("mean_visual",0))/3
    clarity_score = linguistic_res.get("clarity",0)
    creativity_score = visual_res.get("mean_visual",0) * 0.7 + (1 - abs(1 - linguistic_res.get("avg_word_len",4)/5))*0.3
    # brand coherence: use clip similarity if available
    brand_coherence = 0.5 + 0.5*min(1, clarity_score + (visual_res.get("mean_text_similarity") or 0))
    global_score = (emotion_score * w.get("emotion",0.4)
                    + brand_coherence * w.get("brand_coherence",0.3)
                    + creativity_score * w.get("creativity",0.2)
                    + clarity_score * w.get("clarity",0.1))
    out = {
        "emotion_score": utils.normalize_score(emotion_score),
        "clarity_score": utils.normalize_score(clarity_score),
        "creativity_score": utils.normalize_score(creativity_score),
        "brand_coherence_score": utils.normalize_score(brand_coherence),
        "global_score": utils.normalize_score(global_score)
    }
    recs = []
    if out["clarity_score"] < 60:
        recs.append("Raccourcir le message / clarifier l'appel à l'action (CTA).")
    if out["creativity_score"] < 60:
        recs.append("Renforcer l'élément visuel clé (couleurs, contraste, scène mémorable).")
    if out["emotion_score"] < 60:
        recs.append("Augmenter l'intensité émotionnelle via le ton de la voix ou la musique.")
    if not linguistic_res.get("has_cta", False):
        recs.append("Ajouter un call-to-action explicite (ex: 'Visitez notre site', 'Abonnez-vous').")
    out["recommendations"] = recs
    return out
