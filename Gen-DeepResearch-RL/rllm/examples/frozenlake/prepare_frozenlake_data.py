import numpy as np

from rllm.data.dataset import DatasetRegistry


def prepare_frozenlake_data(train_size=10000, test_size=100):
    """
    Prepare and register FrozenLake datasets for training and testing.

    Args:
        train_size (int): Number of training examples to generate
        test_size (int): Number of test examples to generate

    Returns:
        tuple: (train_dataset, test_dataset)
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate random parameters for train and test sets
    train_seeds = np.random.randint(0, 100000, size=train_size)
    test_seeds = np.random.randint(0, 100000, size=test_size)
    train_sizes = np.random.randint(2, 10, size=train_size)
    test_sizes = np.random.randint(2, 10, size=test_size)
    train_ps = np.random.uniform(0.6, 0.85, size=train_size)
    test_ps = np.random.uniform(0.6, 0.85, size=test_size)

    def frozenlake_process_fn(seed, size, p, idx):
        """Process function to create FrozenLake task instances."""
        return {"seed": seed, "size": size, "p": p, "index": idx, "uid": f"{seed}_{size}_{p}"}

    # Create train and test data
    train_data = [frozenlake_process_fn(seed, train_sizes[idx], train_ps[idx], idx) for idx, seed in enumerate(train_seeds)]
    test_data = [frozenlake_process_fn(seed, test_sizes[idx], test_ps[idx], idx) for idx, seed in enumerate(test_seeds)]

    # Register the datasets with the DatasetRegistry
    train_dataset = DatasetRegistry.register_dataset("frozenlake", train_data, "train")
    test_dataset = DatasetRegistry.register_dataset("frozenlake", test_data, "test")

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = prepare_frozenlake_data()
    print(f"Train dataset: {len(train_dataset.get_data())} examples")
    print(f"Test dataset: {len(test_dataset.get_data())} examples")
    print("Sample train example:", train_dataset.get_data()[0])
    print("Sample test example:", test_dataset.get_data()[0])
