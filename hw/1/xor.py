import tensorflow as tf
from tensorflow import keras
import numpy as np


def define_model():
    # Define the model architecture
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_model():
    # Load the quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Define the input and expected output data
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    expected_output = np.array([0, 1, 1, 0], dtype=np.float32)

    # Run inference on the input data and compare the output to the expected output
    num_correct = 0
    for i in range(len(input_data)):
        # Set the input tensor data
        interpreter.set_tensor(input_details[0]['index'], input_data[i:i + 1])

        # Run the model and get the output tensor data
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Compare the output to the expected output
        expected = expected_output[i]
        actual = output_data[0]
        if expected == (actual > 0.5):
            num_correct += 1

    # Calculate the accuracy
    accuracy = num_correct / len(input_data)
    print("Accuracy: ", accuracy)


class XorModel:
    def __init__(self):
        self.model = define_model()
        self.X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [0, 1, 1, 0]

    def train(self):
        self.model.fit(self.X, self.y, epochs=1000, batch_size=4)

    def store(self):
        self.model.save('original_model.tflite')

    def quantize_store(self):
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


if __name__ == '__main__':
    # xor_model = XorModel()
    # xor_model.model = define_model()
    # xor_model.train()
    # xor_model.quantize_store()
    load_model()





