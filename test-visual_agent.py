from agents.visual_agent import analyze_image, aggregate_images_analyses

image_path = "frames_extraites/"

# Analyse d’une seule image
result = analyze_image(image_path + "frame_0.jpg", script_text="Un chat joue dans le jardin")
print(result)

# Analyse de plusieurs images (ex : frames d’une vidéo)
frames = ["frame_000000.jpg", "frame_001000.jpg", "frame_002000.jpg"]
image_paths = [image_path + f for f in frames]
summary = aggregate_images_analyses(image_paths, script_text="Une personne marche dans la rue")
print(summary)