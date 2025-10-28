# predictad/models_open_source.py
# Adaptateurs pour modèles open-source (Whisper, CLIP, BLIP, text classification)
# Si les modèles ne sont pas disponibles, on renvoie des stubs/heuristiques.

import os
import warnings
import librosa


# Tentatives d'import
try:
    from transformers import pipeline, WhisperProcessor, WhisperForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer
    from transformers import CLIPProcessor, CLIPModel
    import torch
except Exception as e:
    warnings.warn("Transformers ou torch non installés ou pas disponibles: fonctionnalités avancées désactivées.")
    pipeline = None
    WhisperProcessor = None
    WhisperForConditionalGeneration = None
    CLIPProcessor = None
    CLIPModel = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None
    torch = None

# Choix de modèles (open-source). Tu peux remplacer par d'autres modèles HF si tu veux.
WHISPER_MODEL = "openai/whisper-small"            # transcription
CLIP_MODEL = "openai/clip-vit-base-patch32"       # embeddings / image-text similarity
BLIP_CAPTION = "Salesforce/blip-image-captioning-base"  # captioning
TEXT_SENTIMENT = "j-hartmann/emotion-english-distilroberta-base"  # emotion detection

def load_whisper_model(device=None):
    if WhisperProcessor is None:
        return None, None
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL)
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL)
    if device is None:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model.to(device)
    return proc, model

def transcribe_audio_whisper(audio_path, device=None):
    """
    Renvoie la transcription texte. Si whisper non disponible, renvoie stub.
    """
    if WhisperProcessor is None:
        return None
    proc, model = load_whisper_model(device=device)
    import soundfile as sf
    import numpy as np
    speech, sr = sf.read(audio_path)
    
    # ✅ Si plusieurs canaux, on passe en   mono
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    # Whisper expects float32
    if speech.dtype != "float32":
        speech = speech.astype("float32")
        
    if sr != 16000:
        print(f"⚠️ Resampling automatique de {sr} Hz vers 16000 Hz...")
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    inputs = proc(speech, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features
    if device and torch:
        input_features = input_features.to(model.device)
    predicted_ids = model.generate(input_features)
    transcription = proc.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def image_caption_blip(image_path, device=None):
    """
    Génère une légende (caption) pour une image via BLIP.
    Si BLIP absent, retourne None.
    """
    try:
        from transformers import BlipProcessor, BlipForConditionalGeneration
    except Exception:
        return None
    if device is None:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    model.to(device)
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    inputs = proc(images=img, return_tensors="pt").to(model.device)
    out = model.generate(**inputs)
    caption = proc.decode(out[0], skip_special_tokens=True)
    return caption


def clip_image_text_similarity(image_path, text, device=None):
    """
    Renvoie une similarité texte-image via CLIP (cosine). Si absent -> None.
    """
    if CLIPProcessor is None or CLIPModel is None:
        return None
    device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained(CLIP_MODEL)
    model = CLIPModel.from_pretrained(CLIP_MODEL).to(device)
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    inputs = proc(text=[text], images=img, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        image_emb = model.get_image_features(**{k:v for k,v in inputs.items() if k.startswith("pixel_values")})
        text_emb = model.get_text_features(**{k:v for k,v in inputs.items() if k.startswith("input_ids") or k.startswith("attention")})
    # normaliser
    image_emb = image_emb / image_emb.norm(p=2, dim=-1, keepdim=True)
    text_emb = text_emb / text_emb.norm(p=2, dim=-1, keepdim=True)
    sim = (image_emb * text_emb).sum().item()
    return float(sim)

def text_emotion_pipeline(text):
    """
    Utilise un modèle de classification émotionnelle (si disponible via transformers pipeline).
    Retour : dict label -> score
    """
    if pipeline is None:
        return None
    try:
        nlp = pipeline("text-classification", model=TEXT_SENTIMENT, return_all_scores=True)
    except Exception:
        # fallback: generic sentiment pipeline
        try:
            nlp = pipeline("sentiment-analysis", return_all_scores=True)
        except Exception:
            return None
    res = nlp(text)
    # pipeline returns list; adapter
    if isinstance(res, list) and len(res)>0:
        # res[0] est une list de labels+scores
        out = {r["label"]: float(r["score"]) for r in (res[0] if isinstance(res[0], list) else res)}
        return out
    return None
