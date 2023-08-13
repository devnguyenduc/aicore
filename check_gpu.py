import tensorflow as tf
# Cấu hình GPU
gpus = tf.config.experimental.list_physical_devices('GPU') # Lấy danh sách GPU có sẵn
tf.config.list_physical_devices('GPU')
print(gpus)

if gpus:
    # Giới hạn bộ nhớ GPU để tránh lỗi ResourceExhaustedError
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
        