from dezero import Model
from dezero.layers import EmbedID
from nlp.sampler import NegativeSamplingLoss
import numpy as np

class CBOW(Model):
    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        super().__init__()
        V, H = vocab_size, hidden_size

        self.in_layers = [EmbedID(V, H) for _ in range(2 * window_size)]
        self.ns_loss = NegativeSamplingLoss(
            np.random.randn(V, H).astype('f') * 0.01,
            corpus,
            power=0.75,
            sample_size=5
        )

        # 임베딩 벡터 하나만 공개용으로 저장 (시각화용 등)
        self.word_vecs = self.in_layers[0].W.data

    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer(contexts[:, i])
        h *= 1 / len(self.in_layers)
        loss = self.ns_loss(h, target)
        return loss