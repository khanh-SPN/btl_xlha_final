import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(data_dir, target_size=(224, 224)):
    data_gen = ImageDataGenerator(
        rescale=1./255,  # Chuẩn hóa giá trị pixel về [0, 1]
        validation_split=0.1  # Dữ liệu validation chiếm 10%
    )
    
    # Dữ liệu huấn luyện
    train_data = data_gen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        batch_size=32,
        class_mode='binary',  # Phân loại nhị phân (chim hoặc cá)
        subset='training'
    )
    
    # Dữ liệu validation
    val_data = data_gen.flow_from_directory(
        os.path.join(data_dir, 'train'),
        target_size=target_size,
        batch_size=32,
        class_mode='binary',  # Phân loại nhị phân
        subset='validation'
    )
    
    return train_data, val_data
