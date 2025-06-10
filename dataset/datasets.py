import numpy as np
import os
import gzip
import urllib.request
from dezero import dataset  # 꼭 있어야 함

# 다운로드 및 해제 관련 유틸
url_base = 'http://yann.lecun.com/exdb/mnist/'
key_file = {
    'train_img': 'train-images-idx3-ubyte.gz',
    'train_label': 'train-labels-idx1-ubyte.gz',
    'test_img': 't10k-images-idx3-ubyte.gz',
    'test_label': 't10k-labels-idx1-ubyte.gz'
}

home = os.path.expanduser("~")
cache_dir = os.path.join(home, ".dezero", "mnist")
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)

def _download(file_name):
    file_path = os.path.join(cache_dir, file_name)
    if not os.path.exists(file_path):
        print(f'Downloading: {file_name}')
        urllib.request.urlretrieve(url_base + file_name, file_path)
    return file_path

def _load_img(file_name):
    file_path = _download(file_name)
    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0

def _load_label(file_name):
    file_path = _download(file_name)
    with gzip.open(file_path, 'rb') as f:
        labels = np.frombuffer(f.read(), np.uint8, offset=8)
    return labels

def get_mnist(train=True):
    if train:
        return _load_img(key_file['train_img']), _load_label(key_file['train_label'])
    else:
        return _load_img(key_file['test_img']), _load_label(key_file['test_label'])

# ✅ DeZero 호환 Dataset 클래스
class MNIST(dataset.Dataset):
    def __init__(self, train=True):
        self.data, self.label = get_mnist(train=train)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return np.array(self.data[index]), np.array(self.label[index])
