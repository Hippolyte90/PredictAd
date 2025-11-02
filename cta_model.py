
from message_gpt import message_gpt4


system_message = """Vous êtes un professionnel de l'analyse des vidéos YouTube
   et devez indiquer si une vidéo ou une publicité contient des phrases incitant à l'action 
   telles que « s'abonner », « acheter maintenant », « cliquez ici », « découvrir », « essayer », « visiter », 
   « contactez-nous », « réserver », « commander », « s'inscrire » ou 
   toute autre phrase similaire encourageant les spectateurs à agir immédiatement ou
   ultérieurement. """
   
   
   
def user_message(text):
    return f"""
   
   Déterminez si le texte fourni contient un appel à l'action.
   Répondez par « Vrai » si le texte contient un appel à l'action   et « Faux » s'il n'en contient pas.
   
Texte: "{text}"
Reasoning:"""

def cta_detect(text):
    prompt = user_message(text)
    text_out = message_gpt4(prompt, system_message).upper()
    print(f"Debug - Model output: {text_out}")
    if "VRAI" in text_out or "YES" in text_out:
        return "True"
    elif "FAUX" in text_out or "NO" in text_out:
        return "False"
    else:
        return "Unknown"

    
    
# Exemple d'utilisation
if __name__ == "__main__":
    text = "N'oubliez pas de vous abonner à notre chaîne pour plus de vidéos incroyables! Visitez notre site web pour en savoir plus."
    role_preds = cta_detect(text)
    print(f"Texte: {text}")
    print(f"Contient un appel à l'action ? {role_preds}")