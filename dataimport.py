import tensorflow as tf
import numpy as np
import create_lstm_model as lstm
import read_video_and_labels as r
import utils

# Khởi tạo TensorFlow session và cấu hình GPU
# Đảm bảo rằng không có phiên làm việc TensorFlow khác đang sử dụng GPU
tf.compat.v1.keras.backend.clear_session()

# Cài đặt giới hạn bộ nhớ GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Thêm GPU vào môi trường đồ họa nhất quán và đặt giới hạn bộ nhớ GPU
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)

        # Xóa biểu đồ tính toán mặc định của TensorFlow và giải phóng bộ nhớ
        tf.compat.v1.reset_default_graph()

    except RuntimeError as e:
        print(e)
# Resource path
resource_path = 'resource/trending/'

# nickname testing
nickname = 'ig_yangg'
nickname_test = 'hahongnguyen'

# Đường dẫn tới file video và file labels của bạn
video_path = resource_path + 'video/' + nickname + '.mp4'
labels_path = resource_path + 'labels/' + nickname + '_labels.csv'

# Đường dẫn tới file video và file labels của bạn
video_path_test = resource_path + 'video/' + nickname_test + '.mp4'
labels_path_test = resource_path + 'labels/' + nickname_test + '_labels.csv'

# Gọi hàm để đọc video và nhãn từ file và chuyển đổi thành X_train và y_train
X_data, y_data = r.read_video_and_labels(video_path, labels_path)

# X_test, y_test = r.read_video_and_labels(video_path_test, labels_path_test)

print(X_data.shape)

# Chuẩn bị dữ liệu và định dạng
X_data_flat = utils.flatten_and_renormalize(X_data)
# X_test_flat = utils.flatten_and_renormalize(X_test)

print(X_data_flat.shape)


# Chuyển đổi X_train_flat thành mảng 3D với time_steps = 64 và input_dim = 1024 * 576 * 3
input_shape = X_data_flat.shape[1:] # Kích thước dữ liệu đầu vào (số khung hình trong chuỗi, chiều rộng khung hình, chiều cao khung hình, số kênh màu)

print(input_shape)

num_classes = y_data.shape[1]
# Tạo mô hình
model = lstm.create_lstm_model(input_shape, num_classes)

# Biên dịch mô hình
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# data là tensor (64, 1769472)
data_split = np.array_split(X_data_flat, 10, axis=0)

# Chuyển các phần dữ liệu thành tensor trên GPU
data_tensors_gpu = [tf.constant(part, dtype=tf.float32) for part in data_split]

# Dữ liệu huấn luyện và nhãn tương ứng
X_train = data_tensors_gpu[:-1]  # 9 phần dữ liệu huấn luyện (sử dụng tất cả ngoại trừ phần cuối cùng)
y_train = y_data[:-1]  # 9 nhãn tương ứng với 9 phần dữ liệu huấn luyện

# Dữ liệu validation và nhãn tương ứng
X_val = data_tensors_gpu[-1]  # Phần cuối cùng dùng làm validation data
y_val = y_data[-1]  # Nhãn tương ứng với phần cuối cùng dùng làm validation data


# Huấn luyện mô hình
# Khởi tạo các tham số huấn luyện
batch_size = 32
epochs = 10
validation_split = 0.2

# history = model.fit(X_train, y_train, epochs=epochs, validation_split=validation_split)
#
# loss, accuracy = model.evaluate(X_val, y_val)
# print(f"Test loss: {loss:.4f}")
# print(f"Test accuracy: {accuracy:.4f}")

# Huấn luyện mô hình trên các phần dữ liệu huấn luyện
for epoch in range(epochs):
     # for i in range(0, len(X_train), batch_size):
     #     print(i)
     X_batch = tf.concat(X_train[0:batch_size], axis=0)
     y_batch = tf.concat(y_train[0:batch_size], axis=0)

     # Train mô hình trên batch hiện tại
     print("train with i: ", i)
     loss = model.train_on_batch(X_batch, y_batch)

     utils.save_model(model, str(epoch) + 'trained_model.h5')

     # Tải mô hình đã lưu
     loaded_model = utils.load_model('resource/aimodel/' + nickname + ".h5")

     # Đánh giá mô hình trên dữ liệu validation sau mỗi epoch
     val_loss = model.evaluate(X_val, y_val, batch_size=batch_size)

     print(f"Epoch {epoch+1}/{epochs} - Training Loss: {loss} - Validation Loss: {val_loss}")

# model.fit(X_train_flat_gpu, np.array(y_train), batch_size=32, epochs=10, validation_split=0.2)

# Lưu mô hình đã huấn luyện


# Đánh giá mô hình trên tập dữ liệu kiểm tra
# loss, accuracy = loaded_model.evaluate(X_test_flat.shape[1:], y_test)

# print(f'Test accuracy: {accuracy}')