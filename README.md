# Snake Species Recognition Project

This project trains and runs a snake image classifier for identifying snake species from camera input or image datasets. It includes a TensorFlow/Keras model, a live camera viewer, and dataset folders for venomous and non-venomous snake groups.

## Project structure

- `converted_keras/train_snake.py` – training pipeline for fine-tuning a pretrained model on the local snake dataset
- `converted_keras/camera_snake.py` – live webcam classifier that loads the trained model and shows predictions in real time
- `converted_keras/keras_model.h5` – base model used for transfer learning
- `converted_keras/snake_model_finetuned.h5` – fine-tuned model output
- `converted_keras/snake_species_finetuned.h5` – final model used by the live demo
- `converted_keras/labels.txt` and `converted_keras/snake_species_labels.txt` – class labels for the model
- `Non-venomous snakes/` – non-venomous snake dataset folders
- `Venomous or potentially medically significant snakes/` – venomous or medically significant snake dataset folders
- `tm-my-image-model/` – exported Teachable Machine model artifacts

## Requirements

- Python 3.10+
- TensorFlow
- OpenCV (`opencv-python`)
- NumPy

A typical environment setup on Windows looks like:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install tensorflow opencv-python numpy
```

If you are on a CPU-only machine, TensorFlow CPU is usually sufficient.

## Training the model

From the project root:

```bash
cd converted_keras
python train_snake.py
```

The training script:

- loads the dataset from the root project folders
- validates image integrity and label consistency
- fine-tunes the base Keras model
- saves the improved model to `snake_species_finetuned.h5`

## Running the live camera classifier

```bash
cd converted_keras
python camera_snake.py
```

This starts a live OpenCV window that:

- opens the default webcam
- captures frames in real time
- preprocesses them for the model
- ranks the top predictions
- displays the current snake class and confidence values

You can pass a different camera index if needed:

```bash
python camera_snake.py --camera 1
```

## Dataset notes

The dataset is organized into species/folder groups under the main snake directories. The training script maps folder names to the model labels and validates that every expected class has readable image samples.

## Model behavior

The project uses a transfer-learning approach:

1. Start from a pretrained Keras image model.
2. Replace or adapt the final classification head to the snake dataset.
3. Train only the needed top layers to preserve the earlier learned visual features.
4. Save the final fine-tuned model for live inference.

## Notes

- Model paths and label files are configured in the scripts, so they work from the project directory structure as-is.
- If your camera is busy or unavailable, try a different `--camera` index.
- If TensorFlow or OpenCV installation fails, ensure your Python version matches the available package versions on your machine.

## License

This project is intended for local research and experimentation. If you plan to reuse or publish it, confirm whether the dataset and model assets have any additional licensing constraints.
