import numpy as np
from dezero import Layer
import dezero.functions as F
import dezero.layers as L

class TimeRNN(Layer):
    def __init__(self, hidden_size, in_size=None, stateful=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.stateful = stateful
        self.rnn_cell = L.RNN(hidden_size, in_size=in_size)  # Layer 기반 RNN 셀 사용
        self.h = None

    def reset_state(self):
        self.h = None
        if hasattr(self.rnn_cell, 'reset_state'):
            self.rnn_cell.reset_state()

    def forward(self, xs):
        N, T, D = xs.shape
        H = self.hidden_size
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        self.layer_list = []

        for t in range(T):
            layer = L.RNN(self.hidden_size, in_size=D)
            layer.x2h.W.data[...] = self.rnn_cell.x2h.W.data
            layer.x2h.b.data[...] = self.rnn_cell.x2h.b.data
            layer.h2h.W.data[...] = self.rnn_cell.h2h.W.data

            if not self.stateful:
                layer.reset_state()
            else:
                layer.h = self.h

            h = layer(xs[:, t, :])
            hs[:, t, :] = h
            self.h = h
            self.layer_list.append(layer)

        return hs

    def backward(self, dhs):
        N, T, H = dhs.shape
        D = self.in_size
        dxs = np.empty((N, T, D), dtype='f')
        dh = 0

        self.rnn_cell.cleargrads()

        for t in reversed(range(T)):
            layer = self.layer_list[t]
            dx = layer.backward(dhs[:, t, :] + dh)
            dh = layer.h2h.backward_grad  # next hidden state's gradient
            dxs[:, t, :] = dx

            self.rnn_cell.x2h.W.grad += layer.x2h.W.grad
            self.rnn_cell.x2h.b.grad += layer.x2h.b.grad
            self.rnn_cell.h2h.W.grad += layer.h2h.W.grad

        return dxs
    
class TimeLSTM(Layer):
    def __init__(self, hidden_size, in_size=None, stateful=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.in_size = in_size
        self.stateful = stateful
        self.lstm_cell = L.LSTM(hidden_size, in_size=in_size)
        self.h, self.c = None, None

    def reset_state(self):
        self.h, self.c = None, None
        if hasattr(self.lstm_cell, 'reset_state'):
            self.lstm_cell.reset_state()

    def forward(self, xs):
        N, T, D = xs.shape
        H = self.hidden_size
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        self.layer_list = []

        for t in range(T):
            layer = L.LSTM(self.hidden_size, in_size=D)
            # 가중치 공유
            for name in ['Wx', 'Wh', 'b']:
                getattr(layer, name).data[...] = getattr(self.lstm_cell, name).data

            h, c = layer(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = h
            self.h, self.c = h, c
            self.layer_list.append(layer)

        return hs

    def backward(self, dhs):
        N, T, H = dhs.shape
        D = self.in_size
        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        self.lstm_cell.cleargrads()

        for t in reversed(range(T)):
            layer = self.layer_list[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx

            # 누적 기울기 (가중치 공유)
            for name in ['Wx', 'Wh', 'b']:
                getattr(self.lstm_cell, name).grad += getattr(layer, name).grad

        return dxs


class TimeEmbedding(Layer):
    def __init__(self, W):
        super().__init__()
        self.embed = L.EmbedID(W.shape[0], W.shape[1])
        self.embed.W.data[...] = W  # 초기값 설정

    def forward(self, xs):
        N, T = xs.shape
        D = self.embed.W.shape[1]
        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = L.EmbedID(self.embed.W.shape[0], self.embed.W.shape[1])
            layer.W.data[...] = self.embed.W.data
            out[:, t, :] = layer(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape
        self.embed.W.grad = np.zeros_like(self.embed.W.data)

        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            self.embed.W.grad += layer.W.grad

        return None
    

from dezero import Layer
import numpy as np
import dezero.layers as L

class TimeLinear(Layer):
    def __init__(self, out_size, in_size=None):
        super().__init__()
        self.linear = L.Linear(out_size, in_size=in_size)
        self.x_shape = None

    def forward(self, x):
        # x: (N, T, D)
        N, T, D = x.shape
        x_reshaped = x.reshape(N * T, D)
        out = self.linear(x_reshaped)  # (N*T, M)
        self.x_shape = x.shape
        return out.reshape(N, T, -1)

    def backward(self, dout):
        # dout: (N, T, M)
        N, T, M = dout.shape
        dout = dout.reshape(N * T, M)
        dx = self.linear.backward(dout)
        return dx.reshape(self.x_shape)

