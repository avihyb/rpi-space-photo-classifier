# Space Engineering Final Project - Space Photo Classifier

## Overview

This project is a complete pipeline for **capturing space images and automatically classifying them** as either "stars" or "Earth horizon" images in order to decide which photos are worth transmitting back to Earth. It was developed as a final project in space engineering, and consists of two main components:

- **PC-side Training Environment:** A Python training pipeline that loads and preprocesses labeled image datasets (good vs bad images of stars and horizon), trains two binary image classification models with data augmentation, and exports the trained classifiers as optimized TensorFlow Lite (TFLite) models for deployment. By using a transfer-learning approach with MobileNetV2, the models can distinguish "good" photos (clear stars or clear horizon) from "bad" ones (e.g., blurry or no meaningful content).

- **Raspberry Pi-side Deployment Environment:** A runtime that runs on a Raspberry Pi (with Camera Module v2) as the satellite onboard computer. The main program (`main.py`) can be invoked in two modes – one to capture a new image and classify it, and another to compress and transmit accumulated images. The Pi captures photos at 1280×720 resolution using the Pi Camera, filters out blurry images using OpenCV, classifies the remaining images using the trained TFLite models (via `predict.py`), stores the classified images in memory (filesystem), and when commanded, compresses and outputs all stored images as a single byte stream (simulating a downlink transmission).

The goal is to conserve downlink bandwidth by only sending high-quality images of stars or the Earth's horizon, while discarding blurry or irrelevant photos automatically.

## Project Structure

```
.
├── models_training/
│   ├── data/
│   │   ├── horizon/
│   │   │   ├── good/
│   │   │   └── bad/
│   │   └── stars/
│   │       ├── good/
│   │       └── bad/
│   ├── src/
│   │   ├── data_utils.py
│   │   ├── model_utils.py
│   │   ├── train.py
│   └── output_models/
│       ├── horizon_model.tflite
│       └── stars_model.tflite
├── predict.py
├── main.py
├── config.conf
├── images_to_send/
│   ├── horizon/
│   └── stars/
└── requirements.txt
```

- **models_training/** – PC training code (data loading, model training, TFLite export).
- **output_models/** – Contains the exported `.tflite` models ready for deployment.
- **predict.py** – Loads TFLite models via tflite-runtime and provides `predict_stars` and `predict_horizon` functions for inference.
- **main.py** – Entry point on the Raspberry Pi. Supports two modes:
  - `-c`, `--capture`: Capture and classify a single image.
  - `-s`, `--send`: Compress and output all stored images.
- **config.conf** – Persistent configuration file tracking boot count, mission count, and picture count.
- **images_to_send/** – Folders where classified images await transmission.
- **requirements.txt** – List of required Python packages for both PC and Raspberry Pi environments.

## PC Training Environment Setup

1. **Create a virtual environment** (recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install tensorflow scikit-learn numpy
   ```

3. **Prepare the dataset**:  
   Organize images under `models_training/data/` with subfolders:
   ```
   data/horizon/good/
   data/horizon/bad/
   data/stars/good/
   data/stars/bad/
   ```

4. **Run training**:
   ```bash
   python models_training/src/train.py
   ```
   - Trains two binary classifiers using MobileNetV2 backbone.
   - Applies data augmentation: flips, rotations, brightness jitter.
   - Splits data into 15% test, 15% validation, 70% training.
   - Outputs evaluation metrics and saves `horizon_model.tflite` and `stars_model.tflite` to `output_models/`.

## Raspberry Pi Deployment Setup

1. **Enable the Pi Camera**  
   ```bash
   sudo raspi-config
   # Enable Legacy Camera interface under Interface Options (for Bullseye OS).
   ```

2. **Install Python 3 and pip**:
   ```bash
   sudo apt update
   sudo apt install python3-pip
   ```

3. **Install deployment requirements**:
   ```bash
   pip3 install tflite-runtime opencv-python numpy
   ```

4. **Copy TFLite models** from `output_models/` on PC to `output_models/` on Pi:
   ```bash
   scp output_models/*.tflite pi@<PI_IP>:/home/pi/space_photo_classifier/output_models/
   ```

5. **Run in Capture mode**:
   ```bash
   python3 main.py -c
   ```

6. **Run in Send mode**:
   ```bash
   python3 main.py -s
   ```

## Example Usage

Capture a clear image of the horizon:
```bash
$ python3 main.py -c
[Mission] Capture and Classify...
[Info] Image sharp, classified as 'horizon'.
[Info] Saved to images_to_send/horizon/image_1.jpg
```

Compress and send stored images:
```bash
$ python3 main.py -s
[Mission] Compress and Send...
[Data] Combined image bytearray (size bytes):
bytearray(b'...')
[Info] Sent 1 images and cleared folders.
```

## Notes

- **Blurriness Check:** Uses OpenCV’s Laplacian variance; threshold adjustable in `main.py`.
- **Model Inputs:** Images are resized to 224×224 and normalized to [-1,1] for TFLite models.
- **Configurations:** `config.conf` stores counters for persistent state across reboots.
- **Extensibility:** You can swap the backbone (e.g., EfficientNet) or add quantization during TFLite conversion if needed.

---

*Created as part of the Space Engineering final project.*  
