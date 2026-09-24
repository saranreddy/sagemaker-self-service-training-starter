"""Generate synthetic MNIST-like dataset for local testing."""
import numpy as np
from pathlib import Path


def generate_data():
    """Generate synthetic image data similar to MNIST."""
    np.random.seed(42)
    
    def generate_split(n_samples):
        X = np.random.rand(n_samples, 28, 28).astype(np.float32) * 255
        
        for i in range(n_samples):
            digit = i % 10
            center = 14
            size = 8
            
            y_start = max(0, center - size // 2)
            y_end = min(28, center + size // 2)
            x_start = max(0, center - size // 2)
            x_end = min(28, center + size // 2)
            
            X[i, y_start:y_end, x_start:x_end] = 200 + np.random.rand(
                y_end - y_start, x_end - x_start
            ) * 55
        
        y = np.array([i % 10 for i in range(n_samples)])
        
        return X, y
    
    X_train, y_train = generate_split(6000)
    X_val, y_val = generate_split(2000)
    X_test, y_test = generate_split(2000)
    
    for split_name, X_split, y_split in [
        ('train', X_train, y_train),
        ('validation', X_val, y_val),
        ('test', X_test, y_test)
    ]:
        split_dir = Path('data') / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(split_dir / 'X.npy', X_split)
        np.save(split_dir / 'y.npy', y_split)
        
        print(f"Generated {split_name} data: {X_split.shape[0]} samples")
    
    print("\nData generation complete!")
    print("Files saved to data/train/, data/validation/, data/test/")


if __name__ == '__main__':
    generate_data()
