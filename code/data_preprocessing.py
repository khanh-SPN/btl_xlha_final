import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def load_data(data_dir, target_size=(224, 224)):
    """
    Tải ảnh từ thư mục và chuyển thành mảng numpy.
    Tất cả ảnh sẽ được thay đổi kích thước về target_size (mặc định 224x224).
    """
    images = []
    labels = []
    
    # Duyệt qua các thư mục chứa ảnh chim và cá
    for label in ['birds', 'fish']:
        folder_path = os.path.join(data_dir, label)
        for file_name in os.listdir(folder_path):
            if file_name.endswith('.jpg') or file_name.endswith('.png'):
                img_path = os.path.join(folder_path, file_name)
                img = load_img(img_path, target_size=target_size)  # Đổi kích thước ảnh
                img_array = img_to_array(img) / 255.0  # Chuẩn hóa ảnh
                images.append(img_array)
                labels.append(label)
    
    return np.array(images), np.array(labels)

def split_data(images, labels, test_size=0.2, val_size=0.2):
    """
    Chia dữ liệu thành 3 phần: train, validation, test.
    Dữ liệu huấn luyện sẽ chiếm 80%, validation 10%, và test 10%.
    """
    # Chia ra thành train và test
    X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=test_size, stratify=labels)
    
    # Chia phần còn lại thành validation và test
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=val_size / (1 - test_size), stratify=y_temp)
    
    return X_train, X_val, X_test, y_train, y_val, y_test
