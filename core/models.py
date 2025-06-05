import dezero
from dezero import layers

class MLP(dezero.Model):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.fc1 = layers.Linear(hidden_size)  # 첫 번째 Linear 레이어
        self.fc2 = layers.Linear(output_size)  # 두 번째 Linear 레이어
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)  # 활성화 함수로 ReLU 사용
        x = self.fc2(x)
        return x