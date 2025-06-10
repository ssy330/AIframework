import dezero
import dezero.functions as F
from dezero import layers, optimizers
from dezero.datasets import MNIST
from dezero.utils import DataLoader
from core.models import MLP
from training.trainer import Trainer

# 1. 데이터셋
train_set = MNIST(train=True)
test_set = MNIST(train=False)
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

# 2. 모델과 옵티마이저
model = MLP(128, 10)
optimizer = optimizers.SGD(lr=0.01).setup(model)

# 3. Trainer 생성 및 훈련
trainer = Trainer(model, optimizer, train_loader, test_loader, epochs=10)
trainer.fit()
