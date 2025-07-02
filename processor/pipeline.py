import os
import time
from datetime import datetime
import cv2

from camera import camera
from utils import metadata
from .preprocess import is_dark, is_blurry, auto_crop
from .tflite_inference import ImageClassifier

MODEL_PATH = os.path.join('models', 'classifier.tflite')
CAPTURE_INTERVAL = 30  # seconds between captures


class ContinuousProcessor:
    """Continuously capture and classify images."""

    def __init__(self, model_path: str = MODEL_PATH, interval: int = CAPTURE_INTERVAL):
        self.classifier = ImageClassifier(model_path)
        self.interval = interval
        camera.ensure_dirs()

    def _update_metadata(self, fname: str, label: str) -> None:
        meta = metadata.load_meta()
        meta[fname] = {
            'datetime': datetime.now().isoformat(),
            'classification': label,
        }
        metadata.save_meta(meta)

    def process_once(self) -> None:
        fname = f"auto_{camera.timestamp()}.jpg"
        path = camera.capture(fname)
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            os.remove(path)
            return
        if is_dark(img) or is_blurry(img):
            os.remove(path)
            return
        cropped = auto_crop(img)
        if cropped.shape != img.shape:
            cv2.imwrite(path, cropped)
        label = self.classifier.predict(path)
        self._update_metadata(fname, label)
        print(f"{fname}: {label}")

    def run(self) -> None:
        while True:
            self.process_once()
            time.sleep(self.interval)


def main():
    processor = ContinuousProcessor()
    processor.run()


if __name__ == '__main__':
    main()
