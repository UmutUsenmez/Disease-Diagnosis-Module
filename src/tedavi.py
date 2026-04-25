import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# -------------------------------------------------
# 1. PATH VE VERİ YÜKLEME
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)

# Senin oluşturduğumuz V4 dosyasını okuyoruz
CSV_PATH = "train_scenarios_robotanic_v4.csv"
df = pd.read_csv(CSV_PATH)

print("Orijinal Veri Boyutu:", df.shape)

# -------------------------------------------------
# 2. V4 VERİSİNE GÜRÜLTÜ (NOISE) EKLEME
# -------------------------------------------------
# Sabit bir rastgelelik için seed belirliyoruz
np.random.seed(42)

# Veri setimizdeki satır sayısı kadar (450 adet) gürültü üretiyoruz
# loc=0 (ortalama sapma), scale=2.5 (standart sapma şiddeti)
gurultu = np.random.normal(loc=0, scale=7.0, size=df.shape[0])

# V4'teki o "kusursuz" skorların üzerine bu gürültüyü ekliyoruz
df["Toplam_Risk_Skoru"] = df["Toplam_Risk_Skoru"] + gurultu

# Skorlar 100'ü geçmesin veya 0'ın altına düşmesin diye tırpanlıyoruz
df["Toplam_Risk_Skoru"] = np.clip(df["Toplam_Risk_Skoru"], 0, 100)

print("✅ Veriye matematiksel gürültü başarıyla eklendi.")

# -------------------------------------------------
# 3. X VE y AYRIMI
# -------------------------------------------------
X = df[["Tehlike_Katsayisi", "Yayilma_Orani_%"]]
y = df["Toplam_Risk_Skoru"]

# -------------------------------------------------
# 4. TRAIN / TEST SPLIT & MODEL EĞİTİMİ
# -------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# -------------------------------------------------
# 5. DEĞERLENDİRME VE KAYIT
# -------------------------------------------------
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\n--- Model Başarısı ---")
print(f"R^2 Skoru: {r2:.4f} (Gürültü eklendiği için artık 1.0 değil)")
print(f"Ortalama Mutlak Hata (MAE): {mae:.2f} birim sapma")

joblib.dump(model, "linear_risk_model_v4_noisy.pkl")
print("\n✅ Gerçekçi Model 'linear_risk_model_v4_noisy.pkl' olarak kaydedildi.")

# -------------------------------------------------
# 6. GRAFİKSEL ÇIKTI
# -------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

ax1.scatter(y_test, y_pred, alpha=0.6, edgecolors='w', s=80)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=3)
ax1.set_xlabel('Gerçek Risk Skoru (Gürültülü)')
ax1.set_ylabel('Modelin Tahmin Ettiği Skor')
ax1.set_title(f'Gerçek vs Tahmin (R² = {r2:.3f})')

hatalar = y_test - y_pred
sns.histplot(hatalar, kde=True, ax=ax2, color="teal")
ax2.axvline(x=0, color='r', linestyle='--', lw=2)
ax2.set_xlabel('Hata Miktarı')
ax2.set_ylabel('Frekans')
ax2.set_title('Hata Dağılımı (Gaussian Noise Etkisi)')

plt.tight_layout()
plt.show()