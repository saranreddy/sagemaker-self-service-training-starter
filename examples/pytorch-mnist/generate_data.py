"""Generate class-separable synthetic MNIST-like dataset for local testing."""

import numpy as np
from pathlib import Path


def generate_data():
    """Generate synthetic image data with class-separable features."""
    np.random.seed(42)

    def generate_split(n_samples):
        # Create separable synthetic data by class
        # Each class gets distinct features to ensure high accuracy
        X = np.zeros((n_samples, 28, 28), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)

        samples_per_class = n_samples // 10

        for class_idx in range(10):
            start_idx = class_idx * samples_per_class
            end_idx = start_idx + samples_per_class

            for i in range(start_idx, min(end_idx, n_samples)):
                # Create base random noise
                img = np.random.rand(28, 28).astype(np.float32) * 50

                # Add class-specific pattern in different quadrants/positions
                # This ensures classes are separable
                if class_idx == 0:
                    # Top-left bright square
                    img[5:12, 5:12] = 200 + np.random.rand(7, 7) * 55
                elif class_idx == 1:
                    # Top-right vertical line
                    img[5:23, 18:21] = 200 + np.random.rand(18, 3) * 55
                elif class_idx == 2:
                    # Bottom-left square
                    img[16:23, 5:12] = 200 + np.random.rand(7, 7) * 55
                elif class_idx == 3:
                    # Bottom-right square
                    img[16:23, 16:23] = 200 + np.random.rand(7, 7) * 55
                elif class_idx == 4:
                    # Center cross pattern
                    img[10:18, 12:16] = 200 + np.random.rand(8, 4) * 55
                    img[12:16, 10:18] = 200 + np.random.rand(4, 8) * 55
                elif class_idx == 5:
                    # Diagonal top-left to bottom-right
                    for d in range(20):
                        img[5 + d, 5 + d] = 220
                        img[5 + d, 6 + d] = 220
                elif class_idx == 6:
                    # Diagonal bottom-left to top-right
                    for d in range(20):
                        img[22 - d, 5 + d] = 220
                        img[22 - d, 6 + d] = 220
                elif class_idx == 7:
                    # Top horizontal line
                    img[7:10, 5:23] = 200 + np.random.rand(3, 18) * 55
                elif class_idx == 8:
                    # Bottom horizontal line
                    img[18:21, 5:23] = 200 + np.random.rand(3, 18) * 55
                else:  # class_idx == 9
                    # Circle-like pattern in center
                    center_y, center_x = 14, 14
                    for y_coord in range(28):
                        for x_coord in range(28):
                            dist = np.sqrt(
                                (y_coord - center_y) ** 2 + (x_coord - center_x) ** 2
                            )
                            if 6 < dist < 9:
                                img[y_coord, x_coord] = 200 + np.random.rand() * 55

                X[i] = img
                y[i] = class_idx

        return X, y

    # Generate train, validation, and test splits
    X_train, y_train = generate_split(6000)
    X_val, y_val = generate_split(2000)
    X_test, y_test = generate_split(2000)

    for split_name, X_split, y_split in [
        ("train", X_train, y_train),
        ("validation", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_dir = Path("data") / split_name
        split_dir.mkdir(parents=True, exist_ok=True)

        np.save(split_dir / "X.npy", X_split)
        np.save(split_dir / "y.npy", y_split)

        print(f"Generated {split_name} data: {X_split.shape[0]} samples")

    print("\nData generation complete!")
    print("Files saved to data/train/, data/validation/, data/test/")
    print("Classes are now separable - model should achieve >95% accuracy")


if __name__ == "__main__":
    generate_data()
