# predictad/utils.py
import json
import os
from PIL import Image
import numpy as np

def load_config(path="config.json"):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config introuvable: {path}. Copie config.example.json -> config.json et rempli les clefs.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def normalize_score(x):
    # ramène en 0-100
    return max(0, min(100, int(round(x * 100))))

def save_report(report_text, out_path="report.md"):
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    return out_path

def load_image(path):
    return Image.open(path)

def mean(a):
    return sum(a)/len(a) if len(a)>0 else 0
