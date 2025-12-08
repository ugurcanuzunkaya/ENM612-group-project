import unittest
import numpy as np
from sklearn.datasets import make_moons
import sys
import os

# Add src to path to allow import if not installed as package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

try:
    from max_min import MaxMinSeparability
except ImportError:
    # Mock class for initial test run if file doesn't exist yet
    class MaxMinSeparability:
        def __init__(self, n_groups, n_hyperplanes_per_group, n_features):
            self.total_vars = 0
            self.J_groups = []


class TestMaxMinSeparability(unittest.TestCase):
    def setUp(self):
        """Her testten önce çalışır. Toy dataset oluşturur."""
        self.X, self.y = make_moons(n_samples=50, noise=0.1, random_state=42)
        # Sınıf etiketlerini 0/1 yerine A ve B kümeleri gibi düşünelim
        self.A = self.X[self.y == 0]
        self.B = self.X[self.y == 1]

        # Test edilecek hiperparametreler
        self.n_features = 2
        self.n_hyperplanes_per_group = 2  # j
        self.n_groups = 2  # r (Toplam hiperdüzlem l = r * j = 4)

        # Model instance
        self.model = MaxMinSeparability(
            n_groups=self.n_groups,
            n_hyperplanes_per_group=self.n_hyperplanes_per_group,
            n_features=self.n_features,
        )

    def test_hyperparameter_initialization(self):
        """Hiperparametrelerin doğru atanıp atanmadığını test et."""
        expected_total_vars = (self.n_features + 1) * (
            self.n_groups * self.n_hyperplanes_per_group
        )
        self.assertEqual(
            self.model.total_vars,
            expected_total_vars,
            "Toplam değişken sayısı (x ve y) yanlış hesaplandı.",
        )
        self.assertEqual(
            len(self.model.J_groups), self.n_groups, "Grup sayısı (I) yanlış."
        )

    def test_loss_function_non_negativity(self):
        """Hata fonksiyonunun (Eq 31-32) asla negatif olamayacağını test et."""
        # Rastgele bir çözüm vektörü oluştur
        if not hasattr(self.model, "total_vars") or self.model.total_vars == 0:
            self.skipTest("Model not implemented yet")

        random_vars = np.random.randn(self.model.total_vars)
        loss = self.model.objective_function(random_vars, self.A, self.B)
        self.assertGreaterEqual(
            loss, 0.0, "Hata fonksiyonu negatif olamaz (Max[0, ...] yapısı)."
        )

    def test_discrete_gradient_shape(self):
        """Ayrık gradyanın, değişkenlerle aynı boyutta vektör döndürdüğünü test et."""
        if not hasattr(self.model, "total_vars") or self.model.total_vars == 0:
            self.skipTest("Model not implemented yet")

        random_vars = np.random.randn(self.model.total_vars)
        grad = self.model.compute_discrete_gradient(random_vars, self.A, self.B)
        self.assertEqual(
            grad.shape, random_vars.shape, "Gradyan vektörü boyutları uyuşmuyor."
        )

    def test_prediction_shape(self):
        """Tahmin fonksiyonunun girdi boyutu kadar çıktı verip vermediğini test et."""
        if not hasattr(self.model, "total_vars") or self.model.total_vars == 0:
            self.skipTest("Model not implemented yet")

        preds = self.model.predict(self.X, np.zeros(self.model.total_vars))
        self.assertEqual(
            len(preds), len(self.X), "Tahmin sayısı örnek sayısına eşit olmalı."
        )


if __name__ == "__main__":
    unittest.main()
