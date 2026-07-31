from pathlib import Path
import json
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_architectural_style_classifier.keras"
CLASS_PATH = BASE_DIR / "models" / "class_names.json"

print("Loading model...")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

print("\n✅ Model Loaded Successfully!")

print("\nInput Shape :", model.input_shape)
print("Output Shape:", model.output_shape)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")