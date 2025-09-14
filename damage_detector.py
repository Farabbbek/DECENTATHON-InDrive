from ultralytics import YOLO
import torch 

class DamageDetector:

    def __init__(self, model_path):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"--- Инициализация модели. Выбрано устройство: {device} ---")

        try:
            self.model = YOLO(model_path)
            self.model.to(device) # <-- Используем определенное выше устройство
            
            print("Модель успешно загружена:")
            print(f"  - Путь: {model_path}")
            print(f"  - Устройство: {self.model.device}")
            print(f"  - Классы: {self.model.names}")
        except Exception as e:
            print(f"ОШИБКА: Не удалось загрузить модель из {model_path}. Ошибка: {e}")
            raise

    def predict(self, image_path):

        results = self.model(image_path)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf[0]
                
                if confidence < 0.25:
                    continue

                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                
                detections.append({
                    'class_name': class_name,
                    'confidence': float(confidence),
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                    
        return detections