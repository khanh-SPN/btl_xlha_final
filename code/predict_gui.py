import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2

# Tải mô hình đã huấn luyện
def load_trained_model():
    return tf.keras.models.load_model('best_model.keras')

# Xử lý ảnh và dự đoán
def predict_image(image_path, model):
    try:
        # Tải ảnh và chuyển thành định dạng phù hợp cho OpenCV
        img = Image.open(image_path).resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Chuẩn hóa ảnh
        
        # Dự đoán
        prediction = model.predict(img_array)
        
        # Xác suất dự đoán
        confidence = prediction[0][0] if prediction[0][0] > 0.5 else 1 - prediction[0][0]
        result = 'Fish' if prediction[0][0] > 0.5 else 'Bird'
        
        return result, confidence, img, prediction
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None, None, None

# Vẽ khung và in kết quả dự đoán lên ảnh
def draw_bounding_boxes(img, result, confidence):
    try:
        # Chuyển ảnh từ PIL sang OpenCV để dễ dàng vẽ
        img_cv = np.array(img)
        img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

        # Phát hiện các vùng đối tượng (giả sử bạn có các khu vực này từ mô hình phát hiện đối tượng)
        # Dưới đây chỉ là ví dụ, bạn có thể thay đổi logic tìm kiếm các đối tượng
        bounding_boxes = []
        
        # Giả định rằng mô hình có thể phát hiện 2 con chim (hoặc cá) trong ảnh
        if result == 'Bird':
            bounding_boxes.append((50, 50, 150, 150))  # Vùng 1
            bounding_boxes.append((160, 50, 240, 150))  # Vùng 2
        elif result == 'Fish':
            bounding_boxes.append((30, 60, 120, 160))  # Vùng 1

        # Vẽ các khung bao quanh vật thể
        for box in bounding_boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 255), 2)  # Vẽ khung màu vàng
            label = f"{result}: {confidence * 100:.2f}%"
            cv2.putText(img_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  # In label lên trên khung

        # Chuyển đổi ảnh từ OpenCV về PIL để hiển thị trên Tkinter
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        return img_pil
    except Exception as e:
        print(f"Error drawing bounding boxes: {e}")
        return img

# Hàm hiển thị ảnh và kết quả dự đoán
def display_image_with_prediction(file_path, result, confidence):
    try:
        # Đọc và xử lý ảnh để vẽ khung
        img = Image.open(file_path)
        img = draw_bounding_boxes(img, result, confidence)  # Vẽ khung và in kết quả lên ảnh

        img = img.resize((224, 224))  # Resize ảnh nếu cần
        img = ImageTk.PhotoImage(img)

        # Hiển thị ảnh
        panel_image.config(image=img)
        panel_image.image = img

        # Hiển thị kết quả
        result_label.config(text=f"Prediction: {result} with {confidence * 100:.2f}% confidence")
    except Exception as e:
        print(f"Error displaying image: {e}")

# Hàm tải ảnh từ máy tính
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        result, confidence, _, _ = predict_image(file_path, model)  # Gọi hàm dự đoán
        if result and confidence is not None:
            display_image_with_prediction(file_path, result, confidence)

# Giao diện chính với Tkinter
root = tk.Tk()
root.title("Image Classification")

# Hiển thị ảnh và kết quả
panel_image = tk.Label(root)
panel_image.pack()

result_label = tk.Label(root, text="Prediction will be shown here")
result_label.pack()

# Nút để tải ảnh
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack()

# Load mô hình đã huấn luyện
model = load_trained_model()

root.mainloop()
