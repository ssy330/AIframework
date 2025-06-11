import os
import numpy as np
import urllib.request
import pickle
from dezero.datasets import Dataset  # 너가 이미 만든 Dataset 기반이라고 가정

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {'train': 'ptb.train.txt', 'valid': 'ptb.valid.txt', 'test': 'ptb.test.txt'}
vocab_file = 'ptb.vocab.pkl'
dataset_dir = os.path.dirname(os.path.abspath(__file__))


def _download(file_name):
    file_path = os.path.join(dataset_dir, file_name)
    if os.path.exists(file_path):
        return
    print('Downloading', file_name)
    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)
    print('Done')


def build_vocab():
    vocab_path = os.path.join(dataset_dir, vocab_file)
    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            return pickle.load(f)

    file_path = os.path.join(dataset_dir, key_file['train'])
    _download(key_file['train'])
    words = open(file_path).read().replace('\n', '<eos>').strip().split()

    word_to_id, id_to_word = {}, {}
    for word in words:
        if word not in word_to_id:
            idx = len(word_to_id)
            word_to_id[word] = idx
            id_to_word[idx] = word

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)
    return word_to_id, id_to_word


class PTB(Dataset):
    def __init__(self, data_type='train', transform=None):
        assert data_type in ('train', 'valid', 'test')
        self.data_type = data_type
        self.word_to_id, self.id_to_word = build_vocab()
        super().__init__(train=(data_type == 'train'), transform=transform)

    def prepare(self):
        file_name = key_file[self.data_type]
        save_path = os.path.join(dataset_dir, f"ptb.{self.data_type}.npy")

        if os.path.exists(save_path):
            self.data = np.load(save_path)
        else:
            _download(file_name)
            words = open(os.path.join(dataset_dir, file_name)).read()
            words = words.replace('\n', '<eos>').strip().split()
            self.data = np.array([self.word_to_id[w] for w in words], dtype=np.int32)
            np.save(save_path, self.data)

        self.label = None  # PTB는 label 없음

    def get_vocab(self):
        return self.word_to_id, self.id_to_word

    def __getitem__(self, index):
        # 텍스트 시퀀스 처리용으로 context-target 쌍을 만들려면 여기서 직접 가공할 수도 있어
        return super().__getitem__(index)
