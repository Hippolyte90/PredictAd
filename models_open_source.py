# predictad/models_open_source.py
# Adaptateurs pour modèles open-source (Whisper, CLIP, BLIP, text classification)
# Si les modèles ne sont pas disponibles, on renvoie des stubs/heuristiques.

import os
import warnings
import librosa
import torch
import soundfile as sf
import numpy as np
warnings.filterwarnings('ignore')


from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h",token=os.getenv("HF_TOKEN"))
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h",token=os.getenv("HF_TOKEN"))

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
WHISPER_MODEL = "openai/whisper-medium"            # transcription
CLIP_MODEL = "openai/clip-vit-base-patch32"       # embeddings / image-text similarity
BLIP_CAPTION = "Salesforce/blip-image-captioning-base"  # captioning
TEXT_SENTIMENT = "j-hartmann/emotion-english-distilroberta-base"  # emotion detection

def load_whisper_model(device=None):
    if WhisperProcessor is None:
        return None, None
    proc = WhisperProcessor.from_pretrained(WHISPER_MODEL,token=os.getenv("HF_TOKEN"))
    model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL,token=os.getenv("HF_TOKEN"))
    if device is None:
        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
    model.to(device)
    return proc, model

def transcribe_audio_whisper(audio_path, device=None, chunk_length_s=30):
    """
    Renvoie la transcription texte avec détection automatique de la langue et gestion améliorée du découpage.
    """
    if WhisperProcessor is None:
        return None
        
    proc, model = load_whisper_model(device=device)
    speech, sr = sf.read(audio_path)
    
    # Conversion en mono si nécessaire
    if speech.ndim > 1:
        speech = np.mean(speech, axis=1)

    # Conversion en float32
    if speech.dtype != "float32":
        speech = speech.astype("float32")
        
    # Resampling à 16kHz si nécessaire
    if sr != 16000:
        speech = librosa.resample(speech, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Calcul de la taille des chunks avec chevauchement
    chunk_size = int(sr * chunk_length_s)
    overlap = int(chunk_size * 0.1)  # 10% de chevauchement
    texts = []

    for start in range(0, len(speech), chunk_size - overlap):
        chunk = speech[start:start + chunk_size]
        
        # Padding si nécessaire pour le dernier chunk
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
            
        inputs = proc(chunk, 
                     sampling_rate=sr, 
                     return_tensors="pt",
                     padding=True)
        
        # Génération avec paramètres optimisés
        generated_ids = model.generate(
            inputs.input_features.to(model.device),
            max_length=448,
            num_beams=5,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            temperature=0.7,
            do_sample=False,
            return_timestamps=True
        )
        
        text = proc.batch_decode(generated_ids, skip_special_tokens=True)[0]
        texts.append(text)

    # Fusion intelligente des segments
    final_text = ""
    for i, text in enumerate(texts):
        if i == 0:
            final_text = text
        else:
            # Éviter les doublons aux jonctions
            overlap_words = 5
            prev_words = final_text.split()[-overlap_words:]
            curr_words = text.split()[:overlap_words]
            
            # Trouver le point de jonction optimal
            for j in range(overlap_words):
                if prev_words[j:] == curr_words[:len(prev_words[j:])]:
                    final_text = final_text + " " + " ".join(text.split()[len(prev_words[j:]):])
                    break
            else:
                final_text = final_text + " " + text

    return final_text.strip()

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
    proc = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", token=os.getenv("HF_TOKEN"))
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", token=os.getenv("HF_TOKEN"))
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
        nlp = pipeline("text-classification", model=TEXT_SENTIMENT, return_all_scores=True, truncation=True,
    max_length=512, token=os.getenv("HF_TOKEN"))
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
