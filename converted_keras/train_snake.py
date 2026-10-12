import os
import stat

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
MODEL_PATH = os.path.join(SCRIPT_DIR, "keras_model.h5")
LABELS_PATH = os.path.join(SCRIPT_DIR, "labels.txt")
OUTPUT_PATH = os.path.join(SCRIPT_DIR, "snake_species_finetuned.h5")
OUTPUT_LABELS_PATH = os.path.join(SCRIPT_DIR, "snake_species_labels.txt")

CLASS_DIRECTORIES = [
    "Non-venomous snakes",
    "Venomous or potentially medically significant snakes",
]
GENERIC_DIRECTORY_LABELS = {
    "All-non-venomous-snake": "Non-venomous snake",
    "All-venomous-snake": "Venomous snake",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.0001
RANDOM_SEED = 123


def load_class_names(path):
    indexed = {}
    with open(path, "r", encoding="utf-8-sig") as labels_file:
        for line in labels_file:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                indexed[int(parts[0])] = parts[1]

    if not indexed or sorted(indexed) != list(range(len(indexed))):
        raise ValueError(f"Class indexes in {path} must start at 0 with no gaps.")
    return [indexed[index] for index in range(len(indexed))]


def is_hidden_or_system(path):
    name = os.path.basename(path)
    if name.startswith((".", "._")):
        return True

    attributes = getattr(os.stat(path, follow_symlinks=False), "st_file_attributes", 0)
    hidden = getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0x2)
    system = getattr(stat, "FILE_ATTRIBUTE_SYSTEM", 0x4)
    return bool(attributes & (hidden | system))


def find_images(directory):
    paths = []
    directories = [directory]

    while directories:
        current = directories.pop()
        for name in os.listdir(current):
            path = os.path.join(current, name)
            if is_hidden_or_system(path):
                continue
            if os.path.isdir(path):
                directories.append(path)
            elif os.path.splitext(name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(path)

    return sorted(paths)


def read_image(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        # imdecode handles Windows paths containing characters cv2.imread cannot decode.
        image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        print(f"Skipping unreadable image: {path}")
        return None

    image = cv2.resize(image, IMAGE_SIZE)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return (image.astype(np.float32) / 127.5) - 1.0


def stratified_indices(class_labels):
    """Keep at least one validation image per class when that class has 2+ images."""
    rng = np.random.default_rng(RANDOM_SEED)
    train_indices, validation_indices = [], []
    for class_index in np.unique(class_labels):
        indices = rng.permutation(np.flatnonzero(class_labels == class_index))
        validation_count = int(round(len(indices) * 0.2)) if len(indices) > 1 else 0
        if len(indices) > 1:
            validation_count = max(1, min(validation_count, len(indices) - 1))
        validation_indices.extend(indices[:validation_count])
        train_indices.extend(indices[validation_count:])

    return np.asarray(train_indices), np.asarray(validation_indices)


def main():
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    class_names = load_class_names(LABELS_PATH)
    class_to_index = {name: index for index, name in enumerate(class_names)}

    images, labels = [], []
    for class_directory in CLASS_DIRECTORIES:
        root = os.path.join(DATASET_PATH, class_directory)
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Dataset class folder was not found: {root}")

        for image_path in find_images(root):
            relative_path = os.path.relpath(image_path, root)
            folder_name = relative_path.split(os.sep, maxsplit=1)[0]
            label_name = GENERIC_DIRECTORY_LABELS.get(folder_name, folder_name)
            if label_name not in class_to_index:
                raise ValueError(
                    f"Dataset folder '{folder_name}' has no matching class in {LABELS_PATH}."
                )

            image = read_image(image_path)
            if image is None:
                continue
            images.append(image)
            labels.append(class_to_index[label_name])

    images = np.asarray(images, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)
    if len(np.unique(labels)) != len(class_names):
        missing = [class_names[i] for i in range(len(class_names)) if not np.any(labels == i)]
        raise ValueError(f"No readable training images for model classes: {missing}")

    train_indices, validation_indices = stratified_indices(labels)
    x_train, y_train = images[train_indices], labels[train_indices]
    x_validation, y_validation = images[validation_indices], labels[validation_indices]
    print(
        f"Loaded {len(images)} images across {len(class_names)} classes; "
        f"training={len(x_train)}, validation={len(x_validation)}."
    )

    base_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    output_count = base_model.output_shape[-1]
    if output_count != len(class_names):
        raise ValueError(
            f"Base model has {output_count} outputs but {LABELS_PATH} has "
            f"{len(class_names)} class names."
        )

    baseline_predictions = base_model.predict(
        x_validation, batch_size=BATCH_SIZE, verbose=0
    )
    baseline_accuracy = float(
        np.mean(np.argmax(baseline_predictions, axis=1) == y_validation)
    )
    print(f"Base model validation top-1 accuracy: {baseline_accuracy:.1%}")

    # Keep the pretrained image features and 34-class head, adapting the head to
    # the local dataset rather than replacing it with an unrelated binary head.
    feature_extractor, classifier = base_model.layers[-2:]
    feature_extractor.trainable = False
    classifier.trainable = True

    augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.1),
            layers.RandomContrast(0.1),
        ],
        name="training_augmentation",
    )
    features = augmentation(base_model.input)
    features = feature_extractor(features, training=False)
    outputs = classifier(features)
    model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=[
            "accuracy",
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
        ],
    )

    counts = np.bincount(y_train, minlength=len(class_names))
    mean_count = float(np.mean(counts[counts > 0]))
    class_weights = {
        index: min(float(np.sqrt(mean_count / count)), 5.0)
        for index, count in enumerate(counts)
        if count > 0
    }

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )
        ],
    )

    best_top1 = max(history.history["val_accuracy"])
    best_top5 = max(history.history["val_top5_accuracy"])
    print(f"Best validation top-1 accuracy: {best_top1:.1%}")
    print(f"Best validation top-5 accuracy: {best_top5:.1%}")
    if best_top1 > baseline_accuracy:
        model.save(OUTPUT_PATH)
        with open(OUTPUT_LABELS_PATH, "w", encoding="utf-8") as labels_file:
            for class_index, class_name in enumerate(class_names):
                labels_file.write(f"{class_index} {class_name}\n")
        print(f"Improved model saved to {OUTPUT_PATH}")
        print(f"Matching class labels saved to {OUTPUT_LABELS_PATH}")
    else:
        print("Fine-tuning did not improve top-1 accuracy; no model was saved.")


if __name__ == "__main__":
    main()
