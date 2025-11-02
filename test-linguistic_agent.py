
from agents.linguistic_agent import analyze_script

def test_analyze_script():
    script_text = """Bonjour à tous! Aujourd'hui, nous allons découvrir les merveilles de la nature. N'oubliez pas de vous abonner à notre chaîne pour plus de vidéos incroyables! Visitez notre site web pour en savoir plus."""
    text_features = analyze_script(script_text)
    print("=== Script Analysis ===")
    for k, v in text_features.items():
        print(f"{k}: {v}")
        
if __name__ == "__main__":
    test_analyze_script()