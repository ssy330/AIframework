# coding: utf-8
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt

from dezero import DataLoader
from dezero.optimizers import Adam
from nlp.datasets import PTB
from nlp.utils import create_contexts_target
from nlp_examples.cbow import CBOW  

# 하이퍼파라미터 설정
window_size = 5
hidden_size = 100
batch_size = 100
max_epoch = 5

# 1. PTB 데이터 로드
ptb = PTB(data_type='train')
corpus = ptb.data
word_to_id, id_to_word = ptb.get_vocab()
vocab_size = len(word_to_id)
print(f"[INFO] 전체 토큰 수: {len(corpus)}")

# 2. 학습 데이터 생성
contexts, target = create_contexts_target(corpus, window_size)
print(f"[INFO] 학습 샘플 수: {len(contexts)}")
dataset = list(zip(contexts, target))
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 3. 모델 및 옵티마이저 초기화
model = CBOW(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam().setup(model)

# 4. 학습 루프
loss_list = []
print("[INFO] 학습 시작")
for epoch in range(max_epoch):
    total_loss = 0
    loss_count = 0
    start_time = time.time()

    for i, (x_batch, t_batch) in enumerate(train_loader):
        loss = model(x_batch, t_batch)  # ⬅️ forward(contexts, target)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        total_loss += float(loss.data)
        loss_count += 1

        if i % 10 == 0:
            print(f"[epoch {epoch+1}] batch {i+1}: loss {loss.data:.4f}")

    avg_loss = total_loss / loss_count
    elapsed_time = time.time() - start_time
    print(f'| epoch {epoch+1} | time {elapsed_time:.1f}s | loss {avg_loss:.4f}')
    loss_list.append(avg_loss)

print("[INFO] 학습 완료")

# 5. 손실 그래프 출력
plt.plot(loss_list)
plt.title("CBOW Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

# 6. 단어 벡터 저장
word_vecs = model.word_vecs
params = {
    'word_vecs': word_vecs.astype(np.float16),
    'word_to_id': word_to_id,
    'id_to_word': id_to_word
}
with open('cbow_params.pkl', 'wb') as f:
    pickle.dump(params, f, -1)