import numpy as np
import cv2
from tflite_runtime.interpreter import Interpreter

# Paths to the TFLite model files (make sure these files are present)
STARS_MODEL_PATH   = "output_models/stars_model.tflite"
HORIZON_MODEL_PATH = "output_models/horizon_model.tflite"

# Fixed input size matching your training (224×224)
IMG_SIZE = (224, 224)

def _load_interpreter(model_path):
    # Create the TFLite interpreter and allocate tensors
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Preload both interpreters once
_interpreter_stars   = _load_interpreter(STARS_MODEL_PATH)
_interpreter_horizon = _load_interpreter(HORIZON_MODEL_PATH)

def _run_tflite_inference(interpreter, image: np.ndarray) -> float:
    """Run the interpreter on a single image and return the raw sigmoid score."""
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Preprocess: resize, convert to float32, scale to [-1,1], and add batch dim
    img = cv2.resize(image, IMG_SIZE)
    img = img.astype(np.float32)
    img = (img / 127.5) - 1.0
    img = np.expand_dims(img, axis=0)

    # Set the tensor and invoke
    interpreter.set_tensor(input_details['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details['index'])
    return float(output[0][0])

def predict_stars(image: np.ndarray) -> bool:
    """
    Returns True if the TFLite 'stars' model predicts this image as 'good stars'.
    """
    score = _run_tflite_inference(_interpreter_stars, image)
    return score >= 0.5

def predict_horizon(image: np.ndarray) -> bool:
    """
    Returns True if the TFLite 'horizon' model predicts this image as 'good horizon'.
    """
    score = _run_tflite_inference(_interpreter_horizon, image)
    return score >= 0.5
