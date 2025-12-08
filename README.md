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
Modeli `make_moons` veri seti üzerinde eğitmek ve karar sınırlarını çizdirmek için:
```bash
uv run main.py
```
Bu komut, eğitimi başlatacak ve sonuçta `decision_boundary.png` adında bir görsel oluşturacaktır.

## 📂 Proje Yapısı

```
.
├── src/
│   └── max_min.py       # Algoritmanın ana sınıfı (MaxMinSeparability)
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

`main.py` çalıştırıldığında elde edilen `decision_boundary.png` görseli şunları gösterir:
- **Mavi Noktalar:** A Sınıfı (Min Region)
- **Kırmızı Noktalar:** B Sınıfı (Max Region)
- **Kontur Alanları:** Modelin karar sınırları.

Model, `make_moons` gibi lineer ayrılamayan bir veri setini, birden fazla doğru parçası kullanarak başarıyla ayırmaktadır. Başlangıçta yüksek olan hata değeri (Loss), iterasyonlar ilerledikçe azalmakta ve 0'a yaklaşmaktadır. Bu, algoritmanın yakınsadığını gösterir.

## 🔮 Gelecek Çalışmalar (Future Updates)

Bu proje şu an temel bir implementasyondur. İleride yapılabilecek geliştirmeler:

1.  **Hiperparametre Optimizasyonu:** `n_groups` ve `n_hyperplanes_per_group` parametrelerinin otomatik seçimi için Cross-Validation eklenebilir.
2.  **Daha Hızlı Çözücüler:** Gurobi yerine açık kaynaklı çözücüler (örneğin OSQP veya SciPy) entegre edilerek lisans bağımlılığı azaltılabilir.
3.  **Büyük Veri Desteği:** Kod şu an tüm veriyi bellekte tutmaktadır. Büyük veri setleri için "Mini-batch" yaklaşımı eklenebilir.
4.  **Çoklu Sınıf Desteği:** Şu an sadece ikili sınıflandırma (Binary Classification) yapılmaktadır. One-vs-All yöntemiyle çoklu sınıf desteği getirilebilir.

---
*Bu proje ENM612 dersi kapsamında hazırlanmıştır.*
