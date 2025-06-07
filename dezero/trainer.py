import time
import numpy as np
import matplotlib.pyplot as plt
import dezero.functions as F

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, train_loader, max_epoch=10, eval_interval=20):
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters, (x, t) in enumerate(train_loader):
                y = model(x)
                loss = F.softmax_cross_entropy(y, t)

                model.cleargrads()
                loss.backward()
                optimizer.update()

                total_loss += float(loss.data)
                loss_count += 1

                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print('| 에폭 %d | 반복 %d | 시간 %.1fs | 손실 %.4f' %
                          (self.current_epoch + 1, iters + 1, elapsed_time, avg_loss))
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.title('Training Loss')
        plt.grid(True)
        plt.legend()
        plt.show()