import tensorflow as tf
import cv2
import numpy as np

def read_video_and_labels(video_path, labels_path):
    # Đọc video từ định dạng mp4
    cap = cv2.VideoCapture(video_path)

    # Đọc nhãn từ file labels (ví dụ: file CSV) và chuyển đổi thành dạng one-hot vector
    labels = np.loadtxt(labels_path, delimiter=',', encoding='utf8')  # Điều chỉnh delimiter phù hợp với định dạng file của bạn
    print(labels.shape)

    labels = np.round(labels).astype(int)
    y_train = tf.keras.utils.to_categorical(labels)  # num_classes là số lượng lớp

    # Khởi tạo danh sách để lưu các chuỗi thời gian (video được chia thành chuỗi thời gian nhỏ)
    video_sequences = []

    # Đọc các khung hình từ video và chia thành các chuỗi thời gian nhỏ
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Chia video thành các chuỗi thời gian có độ dài 10 khung hình (tùy chỉnh độ dài chuỗi thời gian theo ý muốn)
        if len(video_sequences) == 0 or len(video_sequences[-1]) == 32:
            video_sequences.append([frame])
        else:
            video_sequences[-1].append(frame)
    # Kiểm tra kích thước các chuỗi thời gian trong danh sách video_sequences
    sequence_lengths = [len(sequence) for sequence in video_sequences]

    # Chuyển đổi danh sách các chuỗi thời gian thành mảng NumPy (X_train)
    max_sequence_length = sequence_lengths[0]
    X_train = np.zeros((len(video_sequences), max_sequence_length, video_sequences[0][0].shape[0], video_sequences[0][0].shape[1], video_sequences[0][0].shape[2]), dtype=np.uint8)

    for i, sequence in enumerate(video_sequences):
        X_train[i, :len(sequence)] = sequence

    # Trả về X_train và y_train
    return X_train, y_train

def read_all_videos_and_labels():
    return