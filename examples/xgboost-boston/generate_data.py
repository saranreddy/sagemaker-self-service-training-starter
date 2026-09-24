"""Generate synthetic Boston Housing-like dataset for local testing."""
import numpy as np
from pathlib import Path
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


def generate_data():
    """Generate synthetic regression data similar to Boston Housing."""
    np.random.seed(42)
    
    X, y = make_regression(
        n_samples=506,
        n_features=13,
        n_informative=10,
        noise=20.0,
        random_state=42
    )
    
    y = (y - y.min()) / (y.max() - y.min()) * 40 + 5
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
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
