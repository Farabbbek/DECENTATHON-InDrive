from flask import Flask, request, jsonify, render_template
from damage_detector import DamageDetector
import os
import cv2
import numpy as np
import requests
from tqdm import tqdm

MODEL_URL = 'https://drive.google.com/uc?export=download&id=15kG8w51RB4VPb9lQk1fXaurltUewvRz2'
MODEL_DIR = 'models'
MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')
UPLOAD_FOLDER = os.path.join('static', 'uploads')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def download_model(url, path, model_dir):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"Скачивание модели... {url}")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as f, tqdm(
        desc=os.path.basename(path),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))
    print("Модель успешно скачана.")

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(MODEL_PATH):
    download_model(MODEL_URL, MODEL_PATH, MODEL_DIR)

try:
    detector = DamageDetector(MODEL_PATH)
except Exception as e:
    print(f"!!! КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить модель по пути '{MODEL_PATH}'")
    print(f"!!! Подробности: {e}")
    detector = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if detector is None:
        return jsonify({'error': 'Модель не загружена. Проверьте консоль сервера.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'Файл не был отправлен'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'Файл не был выбран'}), 400

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:
            detections = detector.predict(filepath)
            
            def analyze_cleanliness_classic(image_path):
                try:
                    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
                    threshold = 80.0
                    status = "Грязный" if laplacian_var >= threshold else "Чистый"
                    print(f"Показатель 'зашумленности' (пыли): {laplacian_var:.2f}. Статус: {status}")
                    return status, laplacian_var
                except Exception as e:
                    print(f"Ошибка в анализе чистоты: {e}")
                    return "Анализ не удался", 0

            cleanliness_status, dust_level = analyze_cleanliness_classic(filepath)
            
            is_dust_override = False
            only_severe_damage_found = len(detections) > 0 and all(d['class_name'] == 'severe damage' for d in detections)
            
            if only_severe_damage_found and dust_level > 150:
                print("!!! ОБНАРУЖЕНА ОШИБКА 'ПЫЛЬ = ПОВРЕЖДЕНИЕ'. Применяем корректировку.")
                is_dust_override = True
                detections = []
            
            quality_score = 100
            damages_found = []
            
            if not is_dust_override:
                for det in detections:
                    if det['class_name'] != 'good_condition':
                        damages_found.append(det)
                        if 'dent' in det['class_name']: quality_score -= 15
                        elif 'scratch' in det['class_name']: quality_score -= 10
                        else: quality_score -= 20
            
            integrity_status = 'Поврежден' if damages_found else 'Целый'
            
            if integrity_status == 'Целый' and cleanliness_status == 'Грязный':
                quality_score = 90
                summary_text = "Автомобиль целый, но требует мойки."
            elif not damages_found:
                quality_score = 100
                summary_text = "Автомобиль в отличном состоянии."
            else:
                if quality_score >= 80: summary_text = "Обнаружены незначительные дефекты."
                elif quality_score >= 50: summary_text = "Обнаружены заметные повреждения."
                else: summary_text = "Автомобиль в плохом состоянии."

            if any(d['class_name'] == 'severe damage' for d in damages_found):
                quality_score = 10
                summary_text = "Автомобиль в плохом состоянии, обнаружены серьезные повреждения."
           
            return jsonify({
                'integrity_status': integrity_status,
                'cleanliness_status': cleanliness_status,
                'quality_score': quality_score,
                'summary_text': summary_text
            })
        except Exception as e:
            return jsonify({'error': f'Ошибка при обработке изображения: {str(e)}'}), 500
if __name__ == '__main__':
    print("--- Запуск сервера детекции повреждений ---")
    if detector:
        print(f"--> Модель успешно загружена из: {MODEL_PATH}")
    app.run(debug=True, port=5000)