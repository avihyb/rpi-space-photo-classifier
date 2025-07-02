"""Simple example to retrain the onboard classifier with new data."""
import argparse
import os
import tensorflow as tf


def retrain(data_dir: str, epochs: int = 5, output: str = 'models/classifier.tflite') -> None:
    img_size = (96, 96)
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=0.2,
    )
    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        subset='training',
    )
    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=32,
        subset='validation',
    )

    base = tf.keras.applications.MobileNetV2(
        input_shape=img_size + (3,), weights=None, classes=train_gen.num_classes
    )
    base.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    base.fit(train_gen, validation_data=val_gen, epochs=epochs)
    os.makedirs('models', exist_ok=True)
    base.save('models/classifier.h5')

    converter = tf.lite.TFLiteConverter.from_keras_model(base)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output, 'wb') as f:
        f.write(tflite_model)
    print(f'TFLite model written to {output}')


def main():
    parser = argparse.ArgumentParser(description='Retrain classifier and export as TFLite')
    parser.add_argument('data_dir', help='Directory with training images organized by class')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--output', default='models/classifier.tflite')
    args = parser.parse_args()
    retrain(args.data_dir, args.epochs, args.output)


if __name__ == '__main__':
    main()
