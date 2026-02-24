# 🌿 Otonom Yaprak Hastalığı Teşhis Sistemi (Autonomous Leaf Disease Diagnosis System)

Bu proje, otonom tarım robotları için geliştirilmiş **Çok Katmanlı (Pipeline)** bir bilgisayarlı görü ve karar destek sisteminin ikinci aşamasıdır. Amacımız; tarlalarda veya seralarda devriye gezen otonom bir sistemin, bitki yapraklarındaki hastalıkları tespit etmesi, hastalığın türüne göre risk analizi yapması ve **sadece gerekli bölgelere, gerekli miktarda müdahale yapılmasını (Hassas Tarım)** sağlayarak kimyasal ilaç kullanımını minimize etmektir.

![Kapak Fotoğrafı](assets/cover.jpg)

## 🚀 Teknolojik Altyapı ve Sistem Mimarisi
* **Yapay Zeka Modeli:** YOLOv11s (Small - Hız ve yüksek doğruluk optimizasyonu)
* **Görüntü İşleme:** OpenCV, Python
* **Donanım Hedefi:** Otonom Zirai İnsansız Kara Araçları (İKA)

## 📊 Model Başarısı ve Performans Metrikleri
Model, karmaşık tarım ortamlarındaki yaprak verileriyle eğitilmiş olup son derece yüksek doğruluk oranlarına ulaşmıştır.

* **Genel mAP50 Skoru:** **%87.5**
* **Ölümcül Hastalık Tespit Oranları:**
  * Mosaic Virus: **%94.0**
  * Late Blight: **%93.9**
  * Spider Mites: **%91.0**
  * Bacterial Spot: **%90.4**

> Detaylı eğitim süreci, Loss eğrileri ve Confusion Matrix (Hata Matrisi) raporlarına `results/` klasörü altından ulaşabilirsiniz.

## 🛠️ Mühendislik Yaklaşımı: Karşılaşılan Sorunlar ve Çözümler

Otonom bir karar mekanizması tasarlanırken algoritmanın zayıf yönleri yazılımsal mimari ile desteklenmiştir:

1. **"Label Noise" ve False-Positive Optimizasyonu:** Eğitim verisindeki eksik etiketlenmiş arka plan sağlıklı yaprakların modeli zehirlediği (False Positive ürettiği) tespit edilmiştir. Otonom robotun asıl amacı hastalığı bulmak olduğundan, "Healthy" (Sağlıklı) sınıfı model eğitiminden tamamen çıkarılarak arka plan (Background) olarak kabul edilmiş ve sistemin mAP skorunda **+%13'lük muazzam bir artış** sağlanmıştır.

2. **Dinamik Güven Eşiği (Dynamic Confidence Thresholding):** *Yellow Leaf Curl Virus (YLCV)* ve *Leaf Miner* gibi hastalıklar, leke (spot) tabanlı değil, morfolojik (şekilsel) bozukluklar içerir. YOLO mimarisi doğası gereği bu tür şekilsel anormalliklerde düşük güven skoru üretebilmektedir. Bunu aşmak için çıkarım (inference) modülünde özel bir "Dinamik Tolerans" yazılmış; zor hastalıklar için eşik değeri `conf=0.25` seviyesine çekilerek robotun bu hastalıkları kaçırma (False Negative) riski yazılımsal olarak sıfırlanmıştır.

## 💻 Kurulum ve Çalıştırma

Projeyi kendi bilgisayarınızda veya sunucunuzda çalıştırmak için aşağıdaki adımları izleyin:

### 1. Repoyu Klonlayın
```bash
git clone https://github.com/UmutUsenmez/autonomous-leaf-disease-diagnosis.git
cd autonomous-leaf-disease-diagnosis
```

### 2. Gerekli Kütüphaneleri Kurun
```bash
pip install -r requirements.txt
```

### 3. Test Görüntüsü ile Modeli Çalıştırın
```bash
python src/detect_disease.py assets/test_yaprak.jpg
```

## 🔮 Gelecek Çalışmalar (Future Work)

Bu görüntü işleme modülünden elde edilen tespit verileri (hastalık türü, güven skoru, hastalığın yaprakta kapladığı alan yüzdesi) sistemin 3. Katmanı olan Makine Öğrenmesi (Risk Karar) Modeline aktarılacaktır. Bu sayede robot sadece teşhis koymakla kalmayacak, hastalığın yayılma hızını hesaplayarak otonom valf/motor kararları alacaktır.
