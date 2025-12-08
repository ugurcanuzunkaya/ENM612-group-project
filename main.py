import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import gurobipy as gp
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from max_min import MaxMinSeparability


def plot_decision_boundary(model, X, y):
    """
    Max-Min karar sınırlarını görselleştirir.
    """
    # Grid oluştur
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

    # Grid üzerindeki tüm noktaları tahmin et
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 8))
    # Kontur çizimi (Decision Boundary)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.contour(xx, yy, Z, colors="k", linewidths=0.5)

    # Gerçek noktaları çiz
    plt.scatter(
        X[y == 0][:, 0],
        X[y == 0][:, 1],
        c="blue",
        edgecolors="k",
        label="Class A (Min Region)",
    )
    plt.scatter(
        X[y == 1][:, 0],
        X[y == 1][:, 1],
        c="red",
        edgecolors="k",
        label="Class B (Max Region)",
    )

    plt.title(f"Max-Min Separability\nGroups (r)={model.r}, Planes/Group (j)={model.j}")
    plt.legend()
    plt.savefig("decision_boundary.png")
    print("Plot saved to decision_boundary.png")


if __name__ == "__main__":
    # 2. Algoritmayı Çalıştır (Toy Dataset: Moons)
    print("\n--- Training Model on 'Moons' Dataset ---")
    X, y = make_moons(n_samples=200, noise=0.15, random_state=42)

    # Model oluştur (r=2 grup, j=2 düzlem => toplam 4 hiperdüzlem)
    # Bu basit bir yapılandırmadır. Karmaşıklığı artırmak için r ve j'yi artırabilirsiniz.
    model = MaxMinSeparability(
        n_groups=3, n_hyperplanes_per_group=2, n_features=2, max_iter=200
    )

    try:
        model.fit(X, y)
        plot_decision_boundary(model, X, y)
    except gp.GurobiError:
        print("\nGurobi lisansı bulunamadı veya hata oluştu.")
        print(
            "Testleri ve kodu inceleyebilirsiniz, Gurobi kurulu bir ortamda çalışacaktır."
        )
