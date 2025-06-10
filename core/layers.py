import numpy as np
import dezero
from dezero import functions as F
from dezero import Layer, Parameter

class Linear(Layer):
    def __init__(self, out_size, nobias=False):
        super().__init__()
        self.out_size = out_size
        self.nobias = nobias
        self.W = None  # 가중치
        self.b = None  # 바이어스 (옵션)

    def forward(self, x):
        in_size = x.shape[1]

        if self.W is None:
            W_data = np.random.randn(in_size, self.out_size).astype(np.float32) * np.sqrt(1 / in_size)
            self.W = Parameter(W_data)
            if not self.nobias:
                self.b = Parameter(np.zeros(self.out_size, dtype=np.float32))

        y = F.matmul(x, self.W)
        if self.b is not None:
            y = y + self.b
        return y
