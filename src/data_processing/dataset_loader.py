
import tensorflow_datasets as tfds
import os

def load_dataset(dataset_name='cifar10', data_dir='data/raw'):
    """Function to load datasets like CIFAR-10"""
    if dataset_name == 'cifar10':
        if not os.path.exists(os.path.join(data_dir, 'cifar10')):
            print("Downloading CIFAR-10 dataset...")
            dataset, info = tfds.load(dataset_name, with_info=True, as_supervised=True, data_dir=data_dir)
            print(f"Downloaded {dataset_name} and saved in {data_dir}.")
        else:
            print(f"{dataset_name} already exists in {data_dir}.")
        return tfds.load(dataset_name, as_supervised=True, data_dir=data_dir)
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")