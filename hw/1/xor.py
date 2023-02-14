import tensorflow as tf
from tensorflow import keras


def define_model():
    # Define the model architecture
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


class XorModel:
    def __init__(self):
        self.model = define_model()
        self.X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [0, 1, 1, 0]

    def train(self):
        self.model.fit(self.X, self.y, epochs=500, batch_size=4)

    def convert(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # Set the optimization flag to "optimization_type.DEFAULT"
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Set the input and output data types to int8
        converter.target_spec.supported_types = [tf.int8]

        converter.representative_dataset = self.representative_dataset

        # Convert the model to a TFLite flatbuffer file
        tflite_model = converter.convert()

        # Save the TFLite model to a file
        with open('quantized_model.tflite', 'wb') as f:
            f.write(tflite_model)

    # Provide a representative dataset for the quantization process
    def representative_dataset(self):
        for i in range(len(self.X)):
            yield [tf.constant(self.X[i], shape=(1, 2), dtype=tf.float32)]











