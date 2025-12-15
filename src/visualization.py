import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class Visualizer:
    """
    Handles visualization of the decision boundaries and data points.
    """

    def __init__(self, results_dir="results"):
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)

    def plot(self, model, X, y, dataset_name):
        """
        Dispatches to 2D or 3D plotting based on feature count.
        """
        n_features = X.shape[1]
        if n_features == 2:
            self.plot_2d(model, X, y, dataset_name)
        elif n_features == 3:
            self.plot_3d(model, X, y, dataset_name)
        else:
            print(
                f"Skipping plot for {n_features}-dimensional data (only 2D/3D supported)."
            )

    def plot_2d(self, model, X, y, dataset_name):
        """
        Plots the decision boundary for 2D datasets.
        """
        print(f"Generating 2D plot for {dataset_name}...")

        # Create grid
        x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
        y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
        xx, yy = np.meshgrid(
            np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05)
        )

        # Predict for all grid points
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plt.figure(figsize=(10, 8))
        # Contour plot (Decision Boundary)
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        plt.contour(xx, yy, Z, colors="k", linewidths=0.5)

        # Plot data points
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

        plt.title(
            f"Max-Min Separability\nDataset: {dataset_name}\nGroups (r)={model.r}, Planes/Group (j)={model.j}"
        )
        plt.legend()

        output_path = os.path.join(
            self.results_dir, f"{dataset_name}_decision_boundary_2d.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")

    def plot_3d(self, model, X, y, dataset_name):
        """
        Plots data points for 3D datasets.
        """
        print(f"Generating 3D plot for {dataset_name}...")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Plot data points
        ax.scatter(
            X[y == 0][:, 0],
            X[y == 0][:, 1],
            X[y == 0][:, 2],
            c="blue",
            edgecolors="k",
            label="Class A (Min Region)",
            s=40,
        )
        ax.scatter(
            X[y == 1][:, 0],
            X[y == 1][:, 1],
            X[y == 1][:, 2],
            c="red",
            edgecolors="k",
            label="Class B (Max Region)",
            s=40,
        )

        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        ax.set_title(f"Max-Min Separability (3D View)\nDataset: {dataset_name}")
        ax.legend()

        output_path = os.path.join(
            self.results_dir, f"{dataset_name}_decision_boundary_3d.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(f"Plot saved to {output_path}")
