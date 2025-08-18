import tensorflow as tf

# 🔹 Confirm GPU
print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))
print("TensorFlow is using:", "GPU" if tf.config.list_physical_devices('GPU') else "CPU")