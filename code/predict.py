import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image

# Hàm dự đoán với xác suất
def predict_image(image_path, model):
    try:
        # Kiểm tra ảnh có tồn tại
        img = Image.open(image_path).resize((224, 224))  # Resize ảnh
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Thêm chiều batch
        img_array = img_array / 255.0  # Chuẩn hóa ảnh

        # Dự đoán xác suất
        prediction = model.predict(img_array)
        
        # Dự đoán class và xác suất
        predicted_class = 'Bird' if prediction[0][0] > 0.5 else 'Fish'
        confidence = prediction[0][0] if predicted_class == 'Bird' else (1 - prediction[0][0])
        
        return predicted_class, confidence, img
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None
