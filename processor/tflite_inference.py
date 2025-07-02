import numpy as np
import cv2

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # fallback to full tensorflow if tflite-runtime not installed
    from tensorflow.lite.python.interpreter import Interpreter

class ImageClassifier:
    """Lightweight TFLite image classifier."""
    def __init__(self, model_path: str, labels=None):
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels = labels or ["horizon", "stars"]
        self.input_height = self.input_details[0]['shape'][1]
        self.input_width = self.input_details[0]['shape'][2]

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Resize and normalize image for inference."""
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_width, self.input_height))
        img = img.astype(np.float32) / 255.0
        return np.expand_dims(img, axis=0)

    def predict(self, image_path: str) -> str:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        input_data = self.preprocess(img)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        class_idx = int(np.argmax(output_data))
        return self.labels[class_idx]
