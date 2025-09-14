from ultralytics import YOLO
import os


DATA_YAML_PATH = os.path.join('data', 'car_damage_yolo_dataset', 'data.yaml') 

def main():
    print("Загрузка...")
    model = YOLO('yolov8n.pt')

    print(f"Начинаем ЭКСПРЕСС-ОБУЧЕНИЕ на данных из: {DATA_YAML_PATH}")
    results = model.train(
        data=DATA_YAML_PATH,
        
      
        epochs=10,                 
        imgsz=416,                 
        amp=True,                  
       


        workers=4,
        batch=16,                  
        cache=True                 
    )
    print("ЭКСПРЕСС-ОБУЧЕНИЕ УСПЕШНО ЗАВЕРШЕНО!")
    print(f"Ваша быстрая модель сохранена здесь: {results.save_dir}/weights/best.pt")

if __name__ == '__main__':
    main()