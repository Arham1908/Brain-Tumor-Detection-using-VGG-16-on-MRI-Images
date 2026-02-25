from flask import Flask, render_template, request, send_from_directory
from tensorflow.keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# Load the trained model
model = load_model('model/model.h5')

# Class labels
class_labels = ['notumor', 'glioma', 'meningioma', 'pituitary']

# Folders
UPLOAD_FOLDER = './uploads'
REPORT_FOLDER = './reports'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(REPORT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['REPORT_FOLDER'] = REPORT_FOLDER

def predict_tumor(image_path):
    img = load_img(image_path, target_size=(128, 128))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    confidence_score = np.max(predictions)
    label = class_labels[predicted_class_index]
    return ("No Tumor" if label == 'notumor' else f"Tumor: {label}", confidence_score)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            result, confidence = predict_tumor(file_path)

            report_filename = filename.rsplit('.', 1)[0] + '_report.txt'
            report_path = os.path.join(app.config['REPORT_FOLDER'], report_filename)

            with open(report_path, 'w') as f:
                f.write("="*42 + "\n")
                f.write("         MRI Brain Tumor Report\n")
                f.write("="*42 + "\n\n")
                f.write(f"Filename      : {filename}\n")
                f.write(f"Prediction    : {result}\n")
                f.write(f"Confidence    : {confidence*100:.2f}%\n")
                f.write(f"Date & Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                f.write("-"*42 + "\n")
                f.write("            Notes:\n")
                f.write("This result is generated using an AI-powered\n")
                f.write("deep learning model trained on MRI images.\n")
                f.write("Please consult a medical professional for\n")
                f.write("further interpretation and diagnosis.\n")
                f.write("-"*42 + "\n\n")
                f.write("        Thank you for using our system!\n")
                f.write("="*42 + "\n")

            print(f"[INFO] File saved at: {file_path}")
            print(f"[INFO] Report generated at: {report_path}")

            return render_template('index.html',
                                   result=result,
                                   confidence=f"{confidence*100:.2f}",
                                   file_path=f'/uploads/{filename}',
                                   report_filename=report_filename)
    return render_template('index.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    print(f"[INFO] Serving uploaded file: {path}")
    return send_from_directory(os.path.abspath(app.config['UPLOAD_FOLDER']), filename)

@app.route('/download-report/<filename>')
def download_report(filename):
    abs_report_folder = os.path.abspath(app.config['REPORT_FOLDER'])
    file_path = os.path.join(abs_report_folder, filename)
    print(f"[INFO] Attempting to send report file: {file_path}")

    if os.path.exists(file_path):
        return send_from_directory(abs_report_folder, filename, as_attachment=True)
    else:
        print("[ERROR] Report file not found!")
        return "Report file not found", 404

if __name__ == '__main__':
    app.run(debug=True)
