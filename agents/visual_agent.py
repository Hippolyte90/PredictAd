# predictad/agents/visual_agent.py
from PIL import Image, ImageStat
import numpy as np
import os
from predictad.models_open_source import image_caption_blip, clip_image_text_similarity

def analyze_image(image_path, script_text=None):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        return {"ok": False, "error": str(e)}
    stat = ImageStat.Stat(img)
    mean_brightness = sum(stat.mean)/3
    var = sum(stat.var)/3
    visual_score = min(1.0, (var / 1000.0))
    width, height = img.size
    aspect = width/height if height!=0 else 1

    # caption via BLIP (si dispo)
    caption = image_caption_blip(image_path)
    text_similarity = None
    if caption and script_text:
        try:
            sim = clip_image_text_similarity(image_path, script_text)
            text_similarity = sim
        except Exception:
            text_similarity = None

    return {
        "mean_brightness": float(mean_brightness),
        "variance": float(var),
        "aspect_ratio": float(aspect),
        "visual_score": float(visual_score),
        "caption": caption,
        "text_similarity": text_similarity
    }

def aggregate_frame_analyses(frame_paths, script_text=None):
    scores = []
    captions = []
    sims = []
    for p in frame_paths:
        try:
            r = analyze_image(p, script_text=script_text)
            if r.get("visual_score") is not None:
                scores.append(r["visual_score"])
            if r.get("caption"):
                captions.append(r["caption"])
            if r.get("text_similarity") is not None:
                sims.append(r["text_similarity"])
        except:
            continue
    return {
        "mean_visual": float(sum(scores)/len(scores)) if scores else 0.0,
        "n_frames": len(scores),
        "captions": captions[:5],
        "mean_text_similarity": float(sum(sims)/len(sims)) if sims else None
    }
