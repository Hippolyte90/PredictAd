# test_synth_agent.py
from agents.synth_agent import compute_scores, generate_report, plot_radar
import pprint

if __name__ == "__main__":
    # Exemple avec tes données réelles
    audio = {
        'emotion_energy': 0.689354351173457,
        'rms': 0.09605575352907181,
        'tempo': 123.046875
    }

    image = {
        'mean_brightness': 57.07,
        'mean_text_similarity': 0.2058
    }

    script = {
        'clarity': 1.0,
        'has_cta': True,
        'joy': 0.484012633562088,
        'emotion_proxy': 0.009006702341139317
    }

    scores = compute_scores(audio, image, script)
    generate_report(scores)
    plot_radar(scores)
