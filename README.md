# RPi Space Photo Classifier

Lightweight CLI for Raspberry Pi Zero 2 to capture single/burst photos, index metadata, run image‑classification stubs, and seamlessly transfer the latest shot to a ground desktop over SSH.

## Table of Contents

* [Features](#features)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Configuration](#configuration)
* [Usage](#usage)

## Features

* **Single & Burst Capture**: Take individual or burst photos via Raspberry Pi Camera.
* **Metadata Indexing**: Automatically record filename, timestamp, and classification status.
* **Stubbed Classification**: Out‑of‑the‑box stubs for three classification tasks: `horizon`, `star`, and `quality`.
* **SSH/SCP Transfer**: One‑click transfer of the latest photo to your desktop over SSH.
* **Modular Design**: Clean separation into `camera`, `utils`, `inference`, `models`, and now `processor` modules.
* **Continuous Pipeline**: `processor/pipeline.py` implements an always-on loop that captures, filters, crops and classifies images using an on-device TFLite model.

## Prerequisites

* **Hardware**

  * Raspberry Pi Zero 2 with Raspberry Pi Camera Module V2 (enabled via `sudo raspi-config`)
  * MicroSD card with Raspberry Pi OS

* **Software**

  * Python 3.7+
  * `pip3`
  * SSH key‑based access configured between Pi and your ground desktop

* **Python Packages**

  * `picamera` or `picamera2`
  * `opencv-python`
  * `numpy`
  * `Pillow`
  * `pandas`
  * `tflite-runtime` (or full `tensorflow` if preferred)

## Installation

```bash
git clone https://github.com/avihyb/rpi-space-photo-classifier.git
cd rpi-space-photo-classifier

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
```

## Configuration

1. **Update Transfer Settings**
   In `capture.py`, set your SSH details:

   ```python
   MAC_USER = "your_username"
   MAC_DOWNLOAD_DIR = "/path/to/download/dir"
   ```
2. **SSH Setup**
   Ensure you can `ssh pi@<pi-ip>` from your desktop without a password (use SSH keys).
3. **Classification Model**
   Replace stub logic in `inference/predict.py` and add your model files to `models/`.

## Usage

```bash
# Capture photos and predict the type
python3 main.py capture and predict

# Train a model for a specific task (horizon, star, or quality)
python3 main.py train --task <horizon|star|quality>

# Evaluate a trained model on test data
python3 main.py evaluate --task <horizon|star|quality>
```

### Continuous Onboard Pipeline

To run the always-on processor that captures, filters and classifies images in a loop:

```bash
python3 processor/pipeline.py
```

The script stores cropped images and metadata in `camera.PHOTO_DIR` and prints the predicted label for each capture.

### Retraining the Model

If you gather new labeled data you can retrain the lightweight MobileNetV2 model and export a quantized TFLite file:

```bash
python3 retrain.py /path/to/training_data --epochs 10 --output models/classifier.tflite
```

The training data directory should contain one subfolder per class (e.g. `Horizon`, `Stars`). The resulting `classifier.tflite` file will be used by the onboard pipeline.
