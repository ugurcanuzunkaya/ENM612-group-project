# Max-Min Separability Projesi

Bu proje, Adil Hoca'nın makalesinde önerilen **Max-Min Separability** algoritmasının Python ile implementasyonunu içerir. Proje, optimizasyon süreçleri için **Gurobi** çözücüsünü kullanır ve çeşitli veri setleri üzerinde sınıflandırma performansı sunar.

## 📋 İçindekiler
- [Proje Hakkında](#proje-hakkında)
- [Kurulum](#kurulum)
  - [uv ile Kurulum (Önerilen)](#uv-ile-kurulum-önerilen)
  - [pip ile Kurulum](#pip-ile-kurulum)
- [Kullanım](#kullanım)
  - [Dataset Seçenekleri](#dataset-seçenekleri)
  - [Komut Satırı Argümanları](#komut-satırı-argümanları)
- [Deneysel Sonuçlar](#deneysel-sonuçlar)
- [Proje Yapısı](#proje-yapısı)

## 🚀 Proje Hakkında
Bu çalışma, lineer olmayan veri setlerini maksimizasyon ve minimizasyon prensiplerine dayalı parçalı lineer hiperdüzlemlerle (piecewise linear hyperplanes) ayırmayı amaçlar.
- **Yöntem**: Discrete Gradient Method (DGM) ve Gurobi (QP Solver).
- **Amaç**: Sınıflandırma hatasını minimize eden hiperdüzlem katsayılarını bulmak.

## 🛠 Kurulum

Projenin çalışması için **Python 3.10+** ve geçerli bir **Gurobi Lisansı** gereklidir.

### uv ile Kurulum (Önerilen)
`uv`, modern ve hızlı bir Python paket yöneticisidir.

1. **Projeyi Klonlayın:**
   ```bash
   git clone <repo-url>
   cd ENM612-group-project
   ```

2. **Bağımlılıkları Yükleyin:**
   ```bash
   uv sync
   ```

3. **Projeyi Çalıştırın:**
   ```bash
   uv run main.py --dataset moons
   ```

### pip ile Kurulum
Standart `pip` aracını kullanmayı tercih ederseniz:

1. **Sanal Ortam Oluşturun (Opsiyonel ama önerilir):**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Mac/Linux
   # .venv\Scripts\activate   # Windows
   ```

2. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install numpy matplotlib gurobipy scikit-learn ucimlrepo
   ```
   *(Not: `requirements.txt` dosyası varsa `pip install -r requirements.txt` komutunu kullanabilirsiniz.)*

3. **Gurobi Lisansını Kontrol Edin:**
   `gurobipy` kütüphanesinin çalışması için lisansınızın aktif olduğundan emin olun.

## 💻 Kullanım

Modeli eğitmek ve sonuçları görmek için `main.py` dosyasını kullanabilirsiniz.

### Temel Komut
```bash
# uv kullanıyorsanız
uv run main.py --dataset [DATASET_NAME]

# pip/python kullanıyorsanız
python main.py --dataset [DATASET_NAME]
```

### Komut Satırı Argümanları

| Argüman | Tip | Varsayılan | Açıklama |
| :--- | :--- | :---: | :--- |
| `--dataset` | `str` | `moons` | Kullanılacak veri seti ismi (Liste aşağıdadır). |
| `--groups` | `int` | `3` | Sınıflandırma için kullanılacak grup sayısı (r). |
| `--planes` | `int` | `2` | Her gruptaki hiperdüzlem sayısı (j). |

**Örnek 1: Moons Veri Seti (Varsayılan Ayarlar)**
```bash
uv run main.py --dataset moons
```

**Örnek 2: Özel Parametrelerle Blobs 3D**
```bash
uv run main.py --dataset blobs_3d --groups 4 --planes 3
```

### Dataset Seçenekleri

Aşağıdaki veri setleri `src/dataset_loader.py` üzerinden desteklenmektedir:

- **Sentetik Veriler (Sklearn):**
  - `moons`: İki yarım ay şeklindeki veri (2D, Lineer Ayrılamaz).
  - `blobs_3d`: 3 boyutlu, 2 merkezli blob verisi (3D Görselleştirme Testi).
  - `breast_cancer`: Sklearn Meme Kanseri veri seti.

- **UCI Machine Learning Repository Verileri:**
  - `wbcd`: Wisconsin Breast Cancer (Diagnosis).
  - `wbcp`: Wisconsin Breast Cancer (Prognosis).
  - `heart`: Cleveland Heart Disease.
  - `votes`: Congressional Voting Records (Kategorik).
  - `ionosphere`: Ionosphere Radar verisi.
  - `liver`: BUPA Liver Disorders.

- **Diğer:**
  - `custom`: Kendi özel veri setinizi eklemek için şablon.

## 📊 Deneysel Sonuçlar

Tüm deneyler **`results/`** klasörüne kaydedilir. Bu klasörde:
- `*.txt`: Eğitim süresi, metrikler ve ağırlık matrisleri.
- `*.png`: 2D ve 3D görselleştirmeler (Sadece uygun boyutlu veriler için).

**Özet Başarım Tablosu:**

| Veri Seti | Kaynak | Özellik Sayısı | Doğruluk (Accuracy) |
| :--- | :--- | :---: | :---: |
| **Blobs 3D** | Sklearn | 3 | **%100.00** |
| **Breast Cancer** | Sklearn | 30 | **%99.30** |
| **WBCD** | UCI | 30 | **%99.12** |
| **Votes** | UCI | 16 | **%99.08** |
| **Moons** | Sklearn | 2 | **%98.00** |
| **Ionosphere** | UCI | 34 | **%98.01** |
| **WBCP** | UCI | 33 | **%94.95** |
| **Heart** | UCI | 13 | **%93.40** |
| **BUPA Liver** | UCI | 5 | **%27.83** |

## 📂 Proje Yapısı

```
.
├── src/
│   ├── max_min.py        # Algoritma Çekirdeği (Model)
│   ├── dataset_loader.py # Veri Yükleme ve Ön İşleme
│   └── visualization.py  # Görselleştirme (Plotting)
├── main.py               # Ana Çalıştırma Dosyası
├── results/              # Çıktı Klasörü (Model çıktıları bulunmaktadır)
├── pyproject.toml        # Proje ve Bağımlılık Ayarları (uv)
└── README.md             # Dokümantasyon
```

---
*Bu proje ENM612 dersi kapsamında geliştirilmiştir.*
