import tensorflow as tf
from tensorflow import keras
import numpy as np
import os

MODEL_TF = 'model'


def define_model():
    # Define the model architecture
    model = keras.Sequential()
    model.add(keras.layers.Dense(8, input_dim=2, activation='relu'))
    model.add(keras.layers.Dense(8, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def load_model(model_path="quantized_model.tflite"):
    # Load the quantized TFLite model
    interpreter = tf.lite.Interpreter(model_path)
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


def check_size():
    size_unquantized = os.path.getsize('original_model.tflite')
    size_quantized = os.path.getsize('quantized_model.tflite')
    print(f"quantized model size: {size_quantized}, unquantized model size: {size_unquantized}")


class XorModel:
    def __init__(self):
        self.model = define_model()
        self.X = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self.y = [0, 1, 1, 0]

    def train(self):
        self.model.fit(self.X, self.y, epochs=500, batch_size=4)

    def store(self):
        self.model.save(MODEL_TF)
        converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_TF)
        model_no_quant_tflite = converter.convert()

        open('original_model.tflite', "wb").write(model_no_quant_tflite)

    def quantize_store(self):
        converter = tf.lite.TFLiteConverter.from_keras_model(self.model)

        # converter.optimizations = [tf.lite.Optimize.DEFAULT]
        # converter.target_spec.supported_types = [tf.int8]
        # converter.representative_dataset = self.representative_dataset
        # tflite_model = converter.convert()

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = self.representative_dataset
        tflite_model = converter.convert()

        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.representative_dataset = self.representative_dataset
        tflite_model = converter.convert()

        with open('quantized_model.tflite', 'wb') as f:
            f.write(tflite_model)

    def representative_dataset(self):
        for i in range(len(self.X)):
            yield [np.array(self.X[i]).astype('float32')]


if __name__ == '__main__':
    xor_model = XorModel()
    xor_model.model = define_model()
    xor_model.train()
    xor_model.quantize_store()
    xor_model.store()
    check_size()
    load_model()
    load_model("original_model.tflite")

