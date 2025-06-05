import numpy as np
import dezero
from dezero import layers
from dezero import optimizers
from dezero.datasets import MNIST
from dezero.utils import *  # 모델 훈련에 유용한 함수들을 가져옵니다.

# 1. 데이터셋 로딩
train_set = MNIST(train=True)
test_set = MNIST(train=False)

# 데이터셋에서 배치 사이즈 설정
batch_size = 64
train_iter = iter(train_set)
test_iter = iter(test_set)

# 2. 모델 정의 (MLP 모델)
class MLP(dezero.Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc1 = layers.Linear(hidden_size)  # 첫 번째 Fully Connected Layer
        self.fc2 = layers.Linear(output_size)  # 두 번째 Fully Connected Layer
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # ReLU 활성화 함수
        x = self.fc2(x)
        return x

# 3. 모델 및 옵티마이저 설정
model = MLP(128, 10)  # 은닉층 크기 128, 출력층 크기 10 (MNIST는 10개의 클래스)
optimizer = optimizers.SGD(lr=0.01).setup(model)  # 옵티마이저 설정 (SGD)

# 4. 손실 함수 정의
def loss_fn(y_pred, y_true):
    return F.softmax_cross_entropy(y_pred, y_true)

# 5. 훈련 루프
epoch = 10  # 에폭 수
for epoch in range(epoch):
    sum_loss = 0
    for x_batch, t_batch in train_iter:
        model.cleargrads()
        y_pred = model(x_batch)
        loss = loss_fn(y_pred, t_batch)
        loss.backward()
        optimizer.update()

        sum_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {sum_loss / len(train_iter)}")

# 6. 모델 평가 (테스트)
correct = 0
total = 0
with dezero.no_grad():
    for x_batch, t_batch in test_iter:
        y_pred = model(x_batch)
        predicted = y_pred.argmax(axis=1)
        correct += (predicted == t_batch).sum().item()
        total += len(t_batch)

accuracy = correct / total
print(f"Test Accuracy: {accuracy * 100:.2f}%")
