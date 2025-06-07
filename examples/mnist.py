import dezero
import dezero.functions as F
import dezero.optimizers as optimizers
from dezero import DataLoader, cuda
from dezero.datasets import MNIST
from dezero.models import Model
import numpy as np
from dezero.trainer import Trainer
from dezero import layers as L
from dezero.models import MLP

# 하이퍼파라미터
max_epoch = 5
batch_size = 100
hidden_size = 1000

# ✅ DataLoader 사용
train_set = MNIST(train=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

# ✅ 모델, 옵티마이저 설정
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = optimizers.Adam().setup(model)
optimizer.add_hook(optimizers.WeightDecay(1e-4))

# ✅ GPU 전송
if cuda.gpu_enable:
    model.to_gpu()
    train_loader.to_gpu()

# ✅ Trainer 실행
trainer = Trainer(model, optimizer)
trainer.fit(train_loader, max_epoch=max_epoch, eval_interval=10)
trainer.plot()