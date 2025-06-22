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
* **Modular Design**: Clean separation into `camera`, `utils`, `inference`, `models`, and `data` modules.

## Prerequisites

* **Hardware**

  * Raspberry Pi Zero 2 with Raspberry Pi Camera Module V2 (enabled via `sudo raspi-config`)
  * MicroSD card with Raspberry Pi OS

* **Software**

  * Python 3.7+
  * `pip3`
  * SSH key‑based access configured between Pi and your ground desktop

* **Python Packages**

  * `picamera` (or `opencv-python` for `--simulate`)
  * `numpy`
  * `Pillow`
  * `pandas`
  * `scikit-learn` (or `tensorflow` / `tflite-runtime` if replacing the stub)

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
