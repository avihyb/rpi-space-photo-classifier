# RPi Space Photo Classifier

Lightweight CLI for Raspberry Pi Zero 2 to capture single/burst photos, index metadata, run image‑classification stubs, and seamlessly transfer the latest shot to a ground desktop over SSH. :contentReference[oaicite:0]{index=0}

## Table of Contents

- [Features](#features)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [Usage](#usage)  
  - [CLI Overview](#cli-overview)  
  - [Capture Mode](#capture-mode)  
  - [Training Models](#training-models)  
  - [Evaluating Models](#evaluating-models)  
- [Project Structure](#project-structure)  
- [Contributing](#contributing)  
- [License](#license)  
- [Contact](#contact)  

## Features

- **Single & Burst Capture**: Take individual or burst photos via Raspberry Pi Camera.  
- **Metadata Indexing**: Automatically record filename, timestamp, and classification status.  
- **Stubbed Classification**: Out‑of‑the‑box stubs for three classification tasks: `horizon`, `star`, and `quality`. :contentReference[oaicite:1]{index=1}  
- **SSH/SCP Transfer**: One‑click transfer of the latest photo to your desktop over SSH.  
- **Modular Design**: Clean separation into `camera`, `utils`, `inference`, `models`, and `data` modules.

## Prerequisites

- **Hardware**  
  - Raspberry Pi Zero 2 with Raspberry Pi Camera Module V2 (enabled via `sudo raspi-config`)  
  - MicroSD card with Raspberry Pi OS  

- **Software**  
  - Python 3.7+  
  - `pip3`  
  - SSH key‑based access configured between Pi and your ground desktop  

- **Python Packages**  
  - `picamera` (or `opencv-python` for `--simulate`)  
  - `numpy`  
  - `Pillow`  
  - `pandas`  
  - `scikit-learn` (or `tensorflow` / `tflite-runtime` if replacing the stub)  

## Installation

```bash
git clone https://github.com/avihyb/rpi-space-photo-classifier.git
cd rpi-space-photo-classifier

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip3 install -r requirements.txt
