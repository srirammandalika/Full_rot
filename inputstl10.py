# import os
# import urllib.request
# import tarfile

# # Define the URL and destination for STL-10 dataset
# url = 'http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz'
# dataset_dir = './dataset/STL10'

# # Create the destination directory if it doesn't exist
# os.makedirs(dataset_dir, exist_ok=True)

# # Define the file path to save the downloaded file
# tar_file_path = os.path.join(dataset_dir, 'stl10_binary.tar.gz')

# # Download the STL-10 dataset
# def download_stl10_dataset(url, dest_path):
#     print(f'Downloading STL-10 dataset from {url}...')
#     urllib.request.urlretrieve(url, dest_path)
#     print(f'Dataset downloaded and saved to {dest_path}.')

# # Extract the downloaded tar.gz file
# def extract_tar_file(tar_file_path, extract_path):
#     print(f'Extracting {tar_file_path}...')
#     with tarfile.open(tar_file_path, 'r:gz') as tar:
#         tar.extractall(path=extract_path)
#     print(f'Dataset extracted to {extract_path}.')

# # Download and extract STL-10 dataset
# if not os.path.exists(tar_file_path):
#     download_stl10_dataset(url, tar_file_path)
# else:
#     print(f'{tar_file_path} already exists. Skipping download.')

# # Extract the dataset
# extract_tar_file(tar_file_path, dataset_dir)

import torch

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA device")
else:
    device = torch.device("cpu")
    print("Using CPU device")
