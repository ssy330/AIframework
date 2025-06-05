import dezero
from dezero import datasets
from dezero.utils import DataLoader

# MNIST 데이터셋 로딩
class MNIST(dezero.dataset.Dataset):
    def __init__(self, train=True, normalize=True):
        self.train = train
        self.normalize = normalize
        self.data, self.target = datasets.get_mnist(self.train)

        if self.normalize:
            self.data = self.data / 255.0  # 데이터를 0~1로 정규화

    def __getitem__(self, index):
        return self.data[index], self.target[index]

    def __len__(self):
        return len(self.data)
