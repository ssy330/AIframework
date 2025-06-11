import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers, cuda
from dezero import Model
import dezero.functions as F
import dezero.layers as L
from dezero import DataLoader
from dezero.datasets import Spiral
from dezero.trainer import Trainer 

# 모델
class TwoLayerNet(Model):
    def __init__(self, hidden_size, out_size):
        super().__init__()
        self.l1 = L.Linear(hidden_size)
        self.l2 = L.Linear(out_size)
        self.bn1 = L.BatchNorm()

    def forward(self, x):
        y = F.sigmoid(self.bn1(self.l1(x)))
        y = self.l2(y)
        return y

# 하이퍼파라미터
max_epoch = 100
batch_size = 30
hidden_size = 10
lr = 1.0

# 데이터셋 로드
train_set = Spiral(train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# 모델, 옵티마이저 설정
model = TwoLayerNet(hidden_size, 3)
optimizer = optimizers.SGD(lr).setup(model)

# GPU 지원
if cuda.gpu_enable:
    model.to_gpu()
    train_loader.to_gpu()

# Trainer 실행
trainer = Trainer(model, optimizer)
trainer.fit(train_loader, max_epoch=max_epoch, eval_interval=10)

# 예측 결과 시각화
x = np.array([example[0] for example in train_set])
t = np.array([example[1] for example in train_set])
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

# GPU 전송 (예측용)
X = cuda.as_gpu(X) if cuda.gpu_enable else X

with dezero.test_mode():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# 데이터 시각화
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40, marker=markers[c], c=colors[c])
plt.show()