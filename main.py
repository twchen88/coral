import tensorflow as tf
from tensorflow import keras
import numpy as np
import tflite_runtime.interpreter as tflite

# load the data
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# [60000, 28, 28], [10000, 28, 28], [60000, ], [10000, ]

x_train = x_train.astype(np.float32)
x_test = x_test.astype(np.float32)
x_train = x_train.reshape(60000, 28 * 28)/255.0
x_test = x_test.reshape(10000, 28 * 28)/255.0

nClass = 10
y_train = keras.utils.to_categorical(y_train, nClass)
y_test = keras.utils.to_categorical(y_test, nClass)

# Helper function to run inference on a TFLite model
def run_tflite_model(tflite_file, test_image_indices):
  global x_test

  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file),
      experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = x_test[test_image_index]
    test_label = y_test[test_image_index]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    
    # reshape test_image
    test_image = tf.reshape(test_image, [1, 784])
    
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    predictions[i] = output.argmax()

  return predictions


# Change this to test a different image
test_image_index = 1

## Helper function to test the models on one image
def test_model(tflite_file, test_image_index, model_type):
  global y_test

  predictions = run_tflite_model(tflite_file, [test_image_index])

  tmp = tf.reshape(x_test[test_image_index], [28, 28])

def getClass(lst):
    return np.argmax(lst)


# Helper function to evaluate a TFLite model on all images
def evaluate_model(tflite_file, model_type):
  global x_test
  global y_test

  test_image_indices = range(x_test.shape[0])
  predictions = run_tflite_model(tflite_file, test_image_indices)
  tmp = [getClass(e) for e in y_test]

  accuracy = (np.sum(tmp == predictions) * 100) / len(x_test)

  print('%s model accuracy is %.4f%% (Number of test samples=%d)' % (
      model_type, accuracy, len(x_test)))

evaluate_model("mnist_model_quant.tflite", model_type="Quantized")