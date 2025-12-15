import numpy as np
from sklearn.datasets import make_moons, load_breast_cancer, make_blobs
from sklearn.preprocessing import StandardScaler


class DatasetLoader:
    """
    Handles loading and preprocessing of datasets.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def load_dataset(self, dataset_name):
        """
        Dispatches to the specific dataset loader based on the name.
        """
        if dataset_name == "moons":
            return self.load_moons()
        elif dataset_name == "breast_cancer":
            return self.load_breast_cancer()
        elif dataset_name == "blobs_3d":
            return self.load_blobs_3d()
        elif dataset_name == "custom":
            return self.load_custom_dataset()
        else:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: moons, breast_cancer, blobs_3d, custom"
            )

    def load_moons(self):
        print("\n--- Loading 'Moons' Dataset ---")
        X, y = make_moons(n_samples=200, noise=0.15, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y

    def load_breast_cancer(self):
        print("\n--- Loading 'Breast Cancer' Dataset ---")
        data = load_breast_cancer()
        X, y = data.data, data.target
        X = self.scaler.fit_transform(X)
        return X, y

    def load_blobs_3d(self):
        print("\n--- Loading 'Blobs 3D' Dataset ---")
        X, y = make_blobs(n_samples=200, centers=2, n_features=3, random_state=42)
        X = self.scaler.fit_transform(X)
        return X, y

    def load_custom_dataset(self):
        """
        TEMPLATE: Use this method to add your own dataset.

        Steps:
        1. Load your data (e.g., from CSV, Excel, or library).
        2. Separate into features (X) and target (y).
        3. Ensure X is a numpy array of shape (n_samples, n_features).
        4. Ensure y is a numpy array of shape (n_samples,).
        5. Apply scaling if necessary.
        6. Return X, y.
        """
        print("\n--- Loading 'Custom' Dataset ---")
        # --- YOUR CODE HERE ---
        # Example:
        # data = pd.read_csv("my_data.csv")
        # X = data.drop("target", axis=1).values
        # y = data["target"].values

        # Placeholder (Replace this with your data)
        X = np.random.randn(100, 5)  # 100 samples, 5 features
        y = np.random.randint(0, 2, 100)  # Binary target

        # Scaling
        X = self.scaler.fit_transform(X)

        return X, y
