# Max-Min Separability Projesi

Bu proje, Adil Hoca'nın makalesinde önerilen **Max-Min Separability** algoritmasının Python ile implementasyonunu içerir. Proje, **Test Driven Development (TDD)** prensiplerine sadık kalınarak geliştirilmiş ve optimizasyon süreçleri için **Gurobi** çözücüsü kullanılmıştır.

## 📋 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Algoritma Detayları](#algoritma-detayları)
- [Sonuçların Analizi](#sonuçların-analizi)
- [Gelecek Çalışmalar](#gelecek-çalışmalar)

## 🚀 Proje Hakkında
Bu çalışma, lineer olmayan veri setlerini (örneğin `make_moons`) ayırmak için parçalı lineer (piecewise linear) hiperdüzlemler kullanan bir sınıflandırma yöntemidir. Yöntem, klasik SVM veya Lojistik Regresyon'dan farklı olarak, her sınıf için birden fazla hiperdüzlem grubu (polyhedral sets) tanımlar ve **Max-Min** mantığıyla en iyi ayrımı yapmaya çalışır.

Türevsiz optimizasyon (Derivative-Free Optimization) yöntemlerinden biri olan **Discrete Gradient Method (DGM)** kullanılmıştır. İniş yönünü bulmak için ise Gurobi ile bir Kuadratik Programlama (QP) alt problemi çözülmektedir.

## 🛠 Kurulum

Bu projeyi çalıştırmak için sisteminizde Python ve Gurobi lisansının yüklü olması gerekir. Proje bağımlılıkları `uv` paket yöneticisi ile yönetilmektedir.

### Adım 1: Projeyi Klonlayın
```bash
git clone <repo-url>
cd ENM612-group-project
```

### Adım 2: Bağımlılıkları Yükleyin
Eğer `uv` yüklü değilse, önce onu yükleyin veya standart `pip` kullanın.
```bash
# uv ile kurulum (Önerilen)
uv sync

# Veya pip ile
pip install numpy matplotlib gurobipy scikit-learn
```

**Önemli Not:** Gurobi lisansınızın versiyonu ile `gurobipy` kütüphanesinin versiyonunun uyumlu olduğundan emin olun. (Bu projede 12.0.3 versiyonu kullanılmıştır).

## 💻 Kullanım

### Testleri Çalıştırma (TDD)
Kodun doğruluğunu teyit etmek için birim testleri çalıştırabilirsiniz:
```bash
uv run pytest tests/test_max_min.py
```
Bu testler; hiperparametrelerin doğruluğunu, kayıp fonksiyonunun negatif olmamasını ve gradyan boyutlarını kontrol eder.

### Modeli Eğitme ve Görselleştirme
Modeli farklı veri setleri üzerinde çalıştırmak için CLI argümanları eklenmiştir.

**Moons Veri Seti (Varsayılan):**
```bash
uv run main.py --dataset moons --groups 3 --planes 2
```

**Breast Cancer Veri Seti:**
```bash
uv run main.py --dataset breast_cancer
```

**Blobs 3D Veri Seti (3D Görselleştirme Testi):**
```bash
uv run main.py --dataset blobs_3d --groups 3 --planes 2
```

**Özel (Custom) Veri Seti:**
1. `src/dataset_loader.py` dosyasındaki `load_custom_dataset` fonksiyonunu düzenleyin.
2. Aşağıdaki komutu çalıştırın:
```bash
uv run main.py --dataset custom
```

Bu komutlar eğitimi başlatacak, başarı oranlarını (Accuracy, F1-Score) ve toplam süreyi raporlayacaktır. 2 boyutlu veri setleri için `decision_boundary.png` görseli oluşturulur.

## 📂 Proje Yapısı

```
.
├── src/
│   ├── max_min.py       # Algoritmanın ana sınıfı (MaxMinSeparability)
│   ├── dataset_loader.py # Veri seti yükleme ve işleme modülü
│   └── visualization.py  # Görselleştirme modülü (2D/3D)
├── tests/
│   └── test_max_min.py  # Birim testler
├── main.py              # Çalıştırma ve görselleştirme betiği
├── pyproject.toml       # Bağımlılık dosyası
└── README.md            # Dokümantasyon
```

## 🧠 Algoritma Detayları

Kodun temel bileşenleri şunlardır:

1.  **Objective Function (Amaç Fonksiyonu):** Makaledeki Denklem 31 ve 32'nin vektörize edilmiş halidir. Hata (Loss) değeri hesaplanırken, doğru sınıflandırılmış ve "güvenli" bölgedeki noktalar için hata 0 kabul edilir (Hinge Loss benzeri yapı).
2.  **Discrete Gradient (Ayrık Gradyan):** Fonksiyonun türevi alınamadığı için (non-smooth), rastgele yönlerdeki değişimlere bakılarak gradyan tahmin edilir (Tanım 2).
3.  **Direction Finding (Yön Bulma):** Elde edilen gradyan demetinin (bundle) konveks zarfında orijine en yakın nokta bulunur. Bu nokta, en dik iniş yönünün tersidir. Bu işlem Gurobi ile çözülür.

## 📊 Sonuçların Analizi

`main.py` çalıştırıldığında sonuçlar `results/` klasörüne kaydedilir:
- **`{dataset}_results.txt`**: Modelin ağırlıkları, biases değerleri ve başarım metrikleri.
- **`{dataset}_decision_boundary_2d.png`**: 2D veri setleri için karar sınırları.
- **`{dataset}_decision_boundary_3d.png`**: 3D veri setleri için 3 boyutlu dağılım.

Örnek Başarımlar:
- **Moons:** ~98.5% Doğruluk
- **Breast Cancer:** ~98.9% Doğruluk
- **Blobs 3D:** ~100% Doğruluk

2 boyutlu veri setleri için oluşturulan görsel şunları gösterir:
- **Mavi Noktalar:** A Sınıfı (Min Region)
- **Kırmızı Noktalar:** B Sınıfı (Max Region)
- **Kontur Alanları:** Modelin karar sınırları.

Model, `make_moons` gibi lineer ayrılamayan bir veri setini, birden fazla doğru parçası kullanarak başarıyla ayırmaktadır. Başlangıçta yüksek olan hata değeri (Loss), iterasyonlar ilerledikçe azalmakta ve 0'a yaklaşmaktadır. Bu, algoritmanın yakınsadığını gösterir.


---
*Bu proje ENM612 dersi kapsamında hazırlanmıştır.*
