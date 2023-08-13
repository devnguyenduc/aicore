import tensorflow as tf


# Định nghĩa mô hình LSTM
# Định nghĩa hàm create_lstm_model
def create_lstm_model(input_shape, num_classes):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(16, input_shape=input_shape, return_sequences=True),
        tf.keras.layers.LSTM(16),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model