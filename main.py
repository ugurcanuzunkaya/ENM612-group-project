import numpy as np

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import gurobipy as gp
import sys
import os
import time
import argparse

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))

from max_min import MaxMinSeparability
from dataset_loader import DatasetLoader
from visualization import Visualizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Max-Min Separability Model Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="moons",
        help="Dataset name: 'moons', 'breast_cancer', 'blobs_3d', or 'custom' (default: moons)",
    )
    parser.add_argument(
        "--groups",
        type=int,
        default=3,
        help="Number of groups (r) (default: 3)",
    )
    parser.add_argument(
        "--planes",
        type=int,
        default=2,
        help="Number of hyperplanes per group (j) (default: 2)",
    )
    args = parser.parse_args()

    start_time = time.time()

    # Load Dataset
    loader = DatasetLoader()
    try:
        X, y = loader.load_dataset(args.dataset)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Model parameters from CLI
    n_features = X.shape[1]
    n_groups = args.groups
    n_hyperplanes_per_group = args.planes

    print("\n--- Model Configuration ---")
    print(f"Dataset: {args.dataset}")
    print(f"Features: {n_features}")
    print(f"Groups (r): {n_groups}")
    print(f"Planes/Group (j): {n_hyperplanes_per_group}")

    model = MaxMinSeparability(
        n_groups=n_groups,
        n_hyperplanes_per_group=n_hyperplanes_per_group,
        n_features=n_features,
        max_iter=200,
    )

    # Create results directory
    os.makedirs("results", exist_ok=True)

    # Initialize Visualizer
    visualizer = Visualizer(results_dir="results")

    try:
        model.fit(X, y)

        # Save results to text file
        results_path = os.path.join("results", f"{args.dataset}_results.txt")
        W, Biases = model._unpack_variables(model.optimized_vars)

        with open(results_path, "w") as f:
            f.write(f"Dataset: {args.dataset}\n")
            f.write(f"Groups (r): {n_groups}\n")
            f.write(f"Planes per Group (j): {n_hyperplanes_per_group}\n")
            f.write(f"Total Execution Time: {time.time() - start_time:.4f} seconds\n")
            f.write("-" * 50 + "\n")
            f.write("Optimized Variables (Flat):\n")
            f.write(np.array2string(model.optimized_vars, separator=", ") + "\n")
            f.write("-" * 50 + "\n")
            f.write("Weights (W):\n")
            f.write(np.array2string(W, separator=", ") + "\n")
            f.write("-" * 50 + "\n")
            f.write("Biases (Gamma):\n")
            f.write(np.array2string(Biases, separator=", ") + "\n")

        print(f"Results saved to {results_path}")

        # Visualization
        visualizer.plot(model, X, y, args.dataset)

        # Evaluation Metrics
        y_pred = model.predict(X)
        print("\n--- Evaluation Metrics ---")
        print(f"Accuracy: {accuracy_score(y, y_pred):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y, y_pred))

        # Append metrics to results file
        with open(results_path, "a") as f:
            f.write("-" * 50 + "\n")
            f.write("Evaluation Metrics\n")
            f.write(f"Accuracy: {accuracy_score(y, y_pred):.4f}\n")
            f.write("Confusion Matrix:\n")
            f.write(np.array2string(confusion_matrix(y, y_pred), separator=", "))

    except gp.GurobiError:
        print("\nGurobi license not found or error occurred.")
        print(
            "You can examine the tests and code; it will run in an environment with Gurobi installed."
        )

    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.4f} seconds")
