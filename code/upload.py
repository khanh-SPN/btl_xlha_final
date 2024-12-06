import os
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from predict import predict_image
from model import build_model

app = Flask(__name__)

# Đường dẫn cho thư mục upload
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'jpg', 'jpeg', 'png'}

model = build_model()
model.load_weights('model_weights.keras')  # Tải trọng số của mô hình đã huấn luyện

def allowed_file(filename):
    """Kiểm tra định dạng file."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        # Dự đoán và phản hồi
        label = predict_image(model, file_path)
        
        return f"Prediction: {label}"

if __name__ == "__main__":
    app.run(debug=True)
