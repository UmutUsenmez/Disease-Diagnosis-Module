import cv2
from ultralytics import YOLO
import sys

MODEL_PATH = 'models/yolo11s_leaf_disease.pt'

def run_inference(image_path):
    try:

        model = YOLO(MODEL_PATH)

        results = model.predict(image_path, imgsz=640, conf=0.25, save=True)

       
        for result in results:
            result.show() 
            hastalik_sayisi = len(result.boxes)
            if hastalik_sayisi > 0:
                print(f"✅ Tespit Tamamlandı! Yaprak üzerinde {hastalik_sayisi} adet bulgu tespit edildi.")
            else:
                print("✅ Tespit Tamamlandı! Yaprak SAĞLIKLI, hastalık bulunamadı.")

    except Exception as e:
        print(f"❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        run_inference(img_path)
    else:
        print("⚠️ Kullanım: python src/detect_disease.py <resim_yolu>")
        print("💡 Örnek: python src/detect_disease.py assets/test_yaprak.jpeg")