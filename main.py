import pycoral
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify
import matplotlib.pyplot as pyplot

model_path = "model.tflite"

# check that tpu is available
print(edgetpu.list_edge_tpus())

# initialize interpreter for tf
interpreter = edgetpu.make_interpreter(model_path)
interpreter.allocate_tensors()

# process image
img = plt.imread("6.jpg")
img = img[:, :, 0]
img = cv2.resize(img, (28, 28))
img = img.reshape(1, 28 * 28)/255
# img = img.astype("float32")

common.set_input(interpreter, img)
interpreter.invoke()
classes = classify.get_classes(interpreter, top_k=1)