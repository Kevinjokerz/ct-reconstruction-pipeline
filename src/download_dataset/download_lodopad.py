import os
from dival import get_standard_dataset

# Configuration - edit these values directly
PROJECT_DATA_DIR = os.path.abspath("../data/prepared/lodopab")  # Where to store the dataset
DATASET_NAME = 'lodopab'
IMPL = 'skimage'


def main():
    # Create data directory
    os.makedirs(PROJECT_DATA_DIR, exist_ok=True)

    # Tell dival to use this location
    os.environ['DIVAL_DATASETS_PATH'] = PROJECT_DATA_DIR

    print(f"Downloading {DATASET_NAME} dataset to: {PROJECT_DATA_DIR}/{DATASET_NAME}")
    print("Dataset size: ~100GB")
    print("This may take 30+ minutes depending on your connection...")
    print("-" * 60)

    # Download the dataset
    dataset = get_standard_dataset(DATASET_NAME, impl=IMPL)

    print("-" * 60)
    print(f"✓ Download complete!")
    print(f"✓ Training samples: {dataset.get_len('train')}")
    print(f"✓ Validation samples: {dataset.get_len('validation')}")
    print(f"✓ Test samples: {dataset.get_len('test')}")
    print(f"✓ Dataset location: {PROJECT_DATA_DIR}/{DATASET_NAME}")


if __name__ == "__main__":
    main()