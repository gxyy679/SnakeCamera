import os
import stat

import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = ".."
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, DATASET_DIR))
MODEL_PATH = os.path.join(SCRIPT_DIR, "keras_model.h5")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "snake_model_finetuned.h5")

CLASS_NAMES = [
    "Non-venomous snakes",
    "Venomous or potentially medically significant snakes",
]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.0001
RANDOM_SEED = 123


def is_hidden_or_system(path):
    name = os.path.basename(path)
    if name.startswith(".") or name.startswith("._"):
        return True

    attributes = os.stat(path, follow_symlinks=False).st_file_attributes
    hidden_attribute = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2)
    system_attribute = getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 0x4)
    return bool(attributes & (hidden_attribute | system_attribute))


def find_images(class_directory):
    image_paths = []
    directories = [class_directory]

    while directories:
        current_directory = directories.pop()
        for name in os.listdir(current_directory):
            path = os.path.join(current_directory, name)
            if is_hidden_or_system(path):
                continue
            if os.path.isdir(path):
                directories.append(path)
            elif os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                image_paths.append(path)

    return sorted(image_paths)


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        # imdecode handles Windows paths containing characters cv2.imread cannot decode.
        encoded_image = np.fromfile(image_path, dtype=np.uint8)
        image = cv2.imdecode(encoded_image, cv2.IMREAD_COLOR)

    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        return None

    image = cv2.resize(image, IMAGE_SIZE)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


images = []
labels = []

for class_index, class_name in enumerate(CLASS_NAMES):
    class_directory = os.path.join(DATASET_PATH, class_name)
    if not os.path.isdir(class_directory):
        raise FileNotFoundError(f"Class folder was not found: {class_directory}")

    class_image_paths = find_images(class_directory)
    if not class_image_paths:
        raise ValueError(f"No supported images found in: {class_directory}")

    loaded_count = 0
    for image_path in class_image_paths:
        image = read_image(image_path)
        if image is None:
            continue

        images.append(image)
        labels.append(class_index)
        loaded_count += 1

    print(f"Loaded {loaded_count} images for class: {class_name}")

if len(set(labels)) != len(CLASS_NAMES):
    raise ValueError("Every class must have at least one readable image.")

images = np.asarray(images, dtype=np.float32) / 255.0
labels = np.asarray(labels, dtype=np.int32)

x_train, x_validation, y_train, y_validation = train_test_split(
    images,
    labels,
    test_size=0.2,
    random_state=RANDOM_SEED,
    stratify=labels,
)

print(f"Training images: {len(x_train)}; validation images: {len(x_validation)}")

print("Loading base model...")
base_model = tf.keras.models.load_model(MODEL_PATH)

# Teachable Machine stores the feature extractor and classifier as nested models.
# Reuse the feature extractor and replace its original classifier with one Dense layer.
feature_extractor = base_model.layers[-2]
feature_extractor.trainable = False
for layer in feature_extractor.layers:
    layer.trainable = False

outputs = layers.Dense(
    len(CLASS_NAMES),
    activation="softmax",
    name="new_output",
)(feature_extractor.output)
model = tf.keras.Model(inputs=feature_extractor.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=["accuracy"],
)

print("Starting fine-tuning...")
model.fit(
    x_train,
    y_train,
    validation_data=(x_validation, y_validation),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
)

model.save(OUTPUT_PATH)
print(f"Fine-tuned model saved to {OUTPUT_PATH}")
