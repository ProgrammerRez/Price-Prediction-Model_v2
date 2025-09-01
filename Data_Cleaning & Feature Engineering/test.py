import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(3, 3).cuda()
    print("Tensor on GPU:", x)
else:
    print("Running on CPU only")


import tensorflow as tf

# Check TensorFlow version
print("TensorFlow version:", tf.__version__)

# List available GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs detected:")
    for gpu in gpus:
        print(" -", gpu)
else:
    print("No GPU detected. TensorFlow will use CPU.")

# Simple GPU test: matrix multiplication
try:
    with tf.device('/GPU:0'):  # Use GPU if available
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    print("Matrix multiplication successful on GPU.")
except RuntimeError as e:
    print("Error during GPU operation:", e)

# Optional: check if CUDA is actually being used
from tensorflow.python.client import device_lib
print("\nAll devices:")
print(device_lib.list_local_devices())
