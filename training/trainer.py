import numpy as np
import dezero
import dezero.functions as F
from dezero import Variable

class Trainer:
    def __init__(self, model, optimizer, train_loader, test_loader=None, epochs=10, verbose=True):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.verbose = verbose

    def fit(self):
        for epoch in range(self.epochs):
            sum_loss = 0
            for x_batch, t_batch in self.train_loader:
                x = Variable(x_batch)
                t = np.array(t_batch)

                y = self.model(x)
                loss = F.softmax_cross_entropy(y, t)

                self.model.cleargrads()
                loss.backward()
                self.optimizer.update()

                sum_loss += loss.data

            if self.verbose:
                avg_loss = sum_loss / len(self.train_loader)
                print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

            if self.test_loader is not None:
                self.evaluate()

    def evaluate(self):
        correct = 0
        total = 0
        with dezero.no_grad():
            for x_batch, t_batch in self.test_loader:
                x = Variable(x_batch)
                y = self.model(x)
                y_pred = y.data.argmax(axis=1)
                correct += np.sum(y_pred == t_batch)
                total += len(t_batch)
        acc = correct / total
        print(f"Test Accuracy: {acc * 100:.2f}%")
