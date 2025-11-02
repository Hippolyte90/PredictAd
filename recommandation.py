import re
from message_gpt import message_gpt4, stream_gpt4

# scores = {
#         "Audio": round(audio_score * 100, 1),
#         "Visuel": round(visual_score * 100, 1),
#         "Linguistique": round(linguistic_score * 100, 1),
#         "Global": round(final_score * 100, 1),
#     }

# ====== Audio recommandations ====== 

system_message_audio = """Vous êtes un professionnel de l'analyse des vidéos YouTube
   et devez indiquer si l'audio associé à une publicité est de bonne qualité ou non et fournir des recommandations pour l'améliorer.
   Vous etes invité à utiliser le meilleure score audio (combinant: Rythme de la vidéo(tempo), Volume perçu(rms), 
   Intensité émotionnelle moyenne(emotion_energy)) des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité audio"""


# Obtenir le meilleur score audio de référence
user_message_audio1 = """Donne moi le meuilleur score audio (combinant: Rythme de la vidéo(tempo), Volume perçu(rms), 
   Intensité émotionnelle moyenne(emotion_energy)) des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité audio.
   Score audio de référence:"""
   
def get_reference_audio_score():
    prompt = user_message_audio1
    text_out = message_gpt4(prompt, system_message_audio)

    # On essaie d'extraire un nombre (ex : 0.85, 1.0, 75, etc.)
    match = re.search(r"[-+]?\d*\.\d+|\d+", text_out)
    
    if match:
        try:
            reference_score = float(match.group())
            return reference_score
        except ValueError:
            print(f"Debug - Conversion failed for extracted number: {match.group()}")
            return None
    else:
        print(f"Debug - No numeric value found in model output: {text_out}")
        return None
  
 
reference_score_audio = get_reference_audio_score() 


if reference_score_audio < 60:
    reference_score_audio = 60  # On fixe un minimum réaliste
print(f"Reference audio score: {reference_score_audio}, avec 60 comme minimum fixé.")

    
# Fournir des recommandations basées sur le score audio

def user_message_audio2(reference_score_audio, audio_score):
    return f"""
   
    Évaluez la qualité audio d'une publicité vidéo en fonction du score audio de référence fourni.    
    Le score audio de référence est un pourcentage entre 0 et 100, où 100 représente une qualité audio excellente.  

    Score audio de référence: {reference_score_audio}%
    Score audio de la vidéo: {audio_score}%

    Fournir au plus deux recommandations spécifiques pour améliorer la qualité audio si le score de la vidéo est inférieur au score de référence.

    Donne les recommandations sous forme de liste avec l'emoji ➡️ au début de chaque recommandation.
    
    Si le score audio de la vidéo est égal ou supérieur au score de référence, indique que l'audio est de bonne qualité et qu'il n'y a pas besoin d'amélioration.
    
    Recommandations:"""


def recommend_audio(reference_score_audio, audio_score):
    prompt = user_message_audio2(reference_score_audio, audio_score)
    recommendations = message_gpt4(prompt, system_message_audio)
    return recommendations


# ====== Visual recommandations ====== 

system_message_visual = """Vous êtes un professionnel de l'analyse des vidéos YouTube
   et devez indiquer si les éléments visuels associés à une publicité sont de bonne qualité ou non et fournir des recommandations pour les améliorer.
   Vous etes invité à utiliser le meilleure score visuel (combinant: Luminosité modérée(mean_brightness), Cohérence texte–image(mean_text_similarity)) 
   des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité visuelle."""
   
# Obtenir le meilleur score visuel de référence
user_message_visual1 = """Donne moi le meuilleur score visuel (combinant: Luminosité modérée(mean_brightness), 
   Cohérence texte–image(mean_text_similarity)) des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité visuelle.
   Score visuel de référence:"""
   

def get_reference_visual_score():
    prompt = user_message_visual1
    text_out = message_gpt4(prompt, system_message_visual)

    # On essaie d'extraire un nombre (ex : 0.85, 1.0, 75, etc.)
    match = re.search(r"[-+]?\d*\.\d+|\d+", text_out)
    
    if match:
        try:
            reference_score = float(match.group())
            return reference_score
        except ValueError:
            print(f"Debug - Conversion failed for extracted number: {match.group()}")
            return None
    else:
        print(f"Debug - No numeric value found in model output: {text_out}")
        return None


reference_visual_score = get_reference_visual_score()

if reference_visual_score < 60:
    reference_visual_score = 60  # On fixe un minimum réaliste
   
print(f"Reference visual score: {reference_visual_score}, avec 60 comme minimum fixé.")

# Fournir des recommandations basées sur le score visuel

def user_message_visual2(reference_visual_score, visual_score):
    return f"""
   
    Évaluez la qualité visuelle d'une publicité vidéo en fonction du score visuel de référence fourni.    
    Le score visuel de référence est un pourcentage entre 0 et 100, où 100 représente une qualité visuelle excellente.  

    Score visuel de référence: {reference_visual_score}%
    Score visuel de la vidéo: {visual_score}%

    Fournir au plus deux recommandations spécifiques pour améliorer la qualité visuelle si le score de la vidéo est inférieur au score de référence.

    Donne les recommandations sous forme de liste avec l'emoji ➡️ au début de chaque recommandation.
    
    Si le score visuel de la vidéo est égal ou supérieur au score de référence, indique que les éléments visuels sont de bonne qualité et qu'il n'y a pas besoin d'amélioration.
    
    Recommandations:"""
    
def recommend_visual(reference_visual_score, visual_score):
    prompt = user_message_visual2(reference_visual_score, visual_score)
    recommendations = message_gpt4(prompt, system_message_visual)
    return recommendations

# ====== Linguistic recommandations ======

system_message_linguistic = """Vous êtes un professionnel de l'analyse des vidéos YouTube
   et devez indiquer si le script associé à une publicité est de bonne qualité ou non et fournir des recommandations pour l'améliorer.
   Vous etes invité à utiliser le meilleure score linguistique (combinant: Clarté du message(clarity), Présence d'un Call to Action(has_cta), 
   Score de joie(joy), Intensité émotionnelle globale(emotion_proxy)) des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité linguistique."""
   
# Obtenir le meilleur score linguistique de référence

user_message_linguistic1 = """Donne moi le meuilleur score linguistique (combinant: Clarté du message(clarity), Présence d'un Call to Action(has_cta), 
   Score de joie(joy), Intensité émotionnelle globale(emotion_proxy)) des publicités Youtube de bonne qualitées comme référence pour évaluer la qualité linguistique.
   Score linguistique de référence:"""
   
   
 
def get_reference_linguistic_score():
    prompt = user_message_linguistic1
    text_out = message_gpt4(prompt, system_message_linguistic)

    # On essaie d'extraire un nombre (ex : 0.85, 1.0, 75, etc.)
    match = re.search(r"[-+]?\d*\.\d+|\d+", text_out)
    
    if match:
        try:
            reference_score = float(match.group())
            return reference_score
        except ValueError:
            print(f"Debug - Conversion failed for extracted number: {match.group()}")
            return None
    else:
        print(f"Debug - No numeric value found in model output: {text_out}")
        return None
 
    
reference_linguistic_score = get_reference_linguistic_score() 

if reference_linguistic_score < 60:
    reference_linguistic_score = 60  # On fixe un minimum réaliste
      
print(f"Reference linguistic score: {reference_linguistic_score}, avec 60 comme minimum fixé.")


# Fournir des recommandations basées sur le score linguistique

def user_message_linguistic2(reference_linguistic_score, linguistic_score, cta):
    return f"""
   
    Évaluez la qualité linguistique d'une publicité vidéo en fonction du score linguistique de référence fourni.    
    Le score linguistique de référence est un pourcentage entre 0 et 100, où 100 représente une qualité linguistique excellente.  

    Score linguistique de référence: {reference_linguistic_score}%
    Score linguistique de la vidéo: {linguistic_score}%

    Fournir au plus deux recommandations spécifiques pour améliorer la qualité linguistique si le score de la vidéo est inférieur au score de référence.
    
    La valeur du Call to Action (CTA) de la vidéo est {cta}.
    
    Si cette valeur est False, inclure une recommandation pour en ajouter un. Dans le cas contraire, ne pas en parler.

    Donne les recommandations sous forme de liste avec l'emoji ➡️ au début de chaque recommandation.
    
    Si le score linguistique de la vidéo est égal ou supérieur au score de référence, indique que le script est de bonne qualité et qu'il n'y a pas besoin d'amélioration.
    
    Recommandations:"""
    

def recommend_linguistic(reference_linguistic_score, linguistic_score, cta):
    prompt = user_message_linguistic2(reference_linguistic_score, linguistic_score, cta)
    recommendations = message_gpt4(prompt, system_message_linguistic)
    return recommendations
# ====== Fonction principale de recommandation ======

def generate_recommendations(video_scores):
    recommendations = {}
    
    # Recommandations audio
    audio_score = video_scores.get("Audio", 0)
    rec_audio = recommend_audio(reference_score_audio, audio_score)
    recommendations["Audio_rec"] = rec_audio
    
    # Recommandations visuelles
    visual_score = video_scores.get("Visuel", 0)
    rec_visual = recommend_visual(reference_visual_score, visual_score)
    recommendations["Visuel_rec"] = rec_visual
    
    # Recommandations linguistiques
    linguistic_score = video_scores.get("Linguistique", 0)
    cta = video_scores.get("cta", False)
    rec_linguistic = recommend_linguistic(reference_linguistic_score, linguistic_score, cta)
    recommendations["Linguistique_rec"] = rec_linguistic
    
    return recommendations