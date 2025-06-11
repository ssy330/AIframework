from dezero import Layer
import numpy as np
import collections
from dezero.layers import EmbeddingDot
from dezero.functions import sigmoid_cross_entropy

class UnigramSampler:
    def __init__(self, corpus, power, sample_size):
        self.sample_size = sample_size
        self.vocab_size = None
        self.word_p = None

        counts = collections.Counter(corpus)
        vocab_size = len(counts)
        self.vocab_size = vocab_size

        self.word_p = np.zeros(vocab_size)
        for i in range(vocab_size):
            self.word_p[i] = counts[i]

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def get_negative_sample(self, target):
        batch_size = target.shape[0]
        negative_sample = np.zeros((batch_size, self.sample_size), dtype=np.int32)

        for i in range(batch_size):
            p = self.word_p.copy()
            p[target[i]] = 0
            p /= p.sum()
            negative_sample[i] = np.random.choice(
                self.vocab_size, self.sample_size, replace=False, p=p
            )

        return negative_sample


class NegativeSamplingLoss(Layer):
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        super().__init__()
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        for i, layer in enumerate(self.embed_dot_layers):
            setattr(self, f"embed_dot_{i}", layer)

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # 양성 샘플
        score = self.embed_dot_layers[0](h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = sigmoid_cross_entropy(score, correct_label)

        # 음성 샘플
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1 + i](h, negative_target)
            loss += sigmoid_cross_entropy(score, negative_label)

        return loss