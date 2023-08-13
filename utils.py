import numpy as np
def split_data_into_samples(data, sample_shape):
    """
    Chia nhỏ mẫu dữ liệu thành các mẫu nhỏ hơn có kích thước sample_shape.

    Parameters:
        data (np.array): Mảng dữ liệu ban đầu có shape (n_videos, n_frames, width, height, n_channels).
        sample_shape (tuple): Kích thước của mẫu mới (n_frames, width, height, n_channels).

    Returns:
        np.array: Mảng dữ liệu chứa các mẫu nhỏ hơn có shape (n_samples, n_frames, width, height, n_channels).
    """
    n_videos, n_frames, width, height, n_channels = data.shape
    n_frames_sample, width_sample, height_sample, n_channels_sample = sample_shape

    samples = []
    for video in data:
        num_samples = n_frames - n_frames_sample + 1
        for i in range(num_samples):
            sample = video[i : i + n_frames_sample, :, :, :]
            samples.append(sample)

    return np.array(samples)


def flatten_and_renormalize(data):
    """
    Flatten chiều dài và chiều rộng thành 1 chiều và renormalize chiều màu sắc trong dữ liệu video.

    Parameters:
        data (np.array): Dữ liệu video có kích thước (n_videos, n_frames, width, height, n_channels).

    Returns:
        np.array: Dữ liệu video mới đã flatten chiều dài và chiều rộng và renormalize chiều màu sắc, có kích thước (n_videos, n_frames, width*height*n_channels).
    """
    n_videos, n_frames, width, height, n_channels = data.shape

    # Phép reshape và flatten dữ liệu
    reshaped_data = data.reshape(n_videos, n_frames, -1)

    print(reshaped_data.shape)
    # Renormalize dữ liệu về khoảng [0, 1]
    renormalized_data = reshaped_data.astype('float32') / 255.0

    return renormalized_data


def save_model(model, model_path):
    """
    Lưu mô hình đã huấn luyện vào một file.

    Tham số:
    model (tf.keras.Model): Mô hình đã huấn luyện.
    model_path (str): Đường dẫn và tên file để lưu mô hình.

    Return:
    None
    """
    model.save(model_path)
    print(f"Mô hình đã được lưu tại {model_path}")

def load_model(model_path):
    """
    Tải mô hình đã lưu từ file.

    Tham số:
    model_path (str): Đường dẫn và tên file chứa mô hình đã lưu.

    Return:
    tf.keras.Model: Mô hình đã tải.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"Mô hình đã được tải từ {model_path}")
    return model