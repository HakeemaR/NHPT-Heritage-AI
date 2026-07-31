from pathlib import Path
import json

import numpy as np
import tensorflow as tf
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "final_architectural_style_classifier.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"

# Put a test building image inside the uploads folder.
IMAGE_PATH = BASE_DIR / "uploads" / "test_building.jpg"

IMAGE_SIZE = (224, 224)


def load_class_names() -> list[str]:
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as file:
        class_names = json.load(file)

    if not isinstance(class_names, list):
        raise ValueError("class_names.json must contain a JSON list.")

    return class_names


def prepare_image(image_path: Path) -> np.ndarray:
    if not image_path.exists():
        raise FileNotFoundError(
            f"Test image not found:\n{image_path}\n\n"
            "Place an image named test_building.jpg inside the uploads folder."
        )

    image = Image.open(image_path).convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = np.asarray(image, dtype=np.float32)

    # Add batch dimension:
    # (224, 224, 3) -> (1, 224, 224, 3)
    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def main() -> None:
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)

    class_names = load_class_names()
    image_array = prepare_image(IMAGE_PATH)

    print("Making prediction...")
    probabilities = model.predict(image_array, verbose=0)[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_class = class_names[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    print("\nPrediction Results")
    print("------------------")
    print(f"Predicted style: {predicted_class}")
    print(f"Confidence     : {confidence:.2f}%")

    print("\nAll class probabilities:")
    for class_name, probability in sorted(
        zip(class_names, probabilities),
        key=lambda item: item[1],
        reverse=True,
    ):
        print(f"{class_name:<35} {probability * 100:6.2f}%")


if __name__ == "__main__":
    main()
    