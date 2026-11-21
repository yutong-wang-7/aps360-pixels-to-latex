"""models.py: the neural networks (same template as Lab 3).

PIPELINE  1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy -> 5 train
          (helpers: symbols, fonts_config, render, augment; train.py uses
           data, models, training, evaluation, plotting, report)
THIS FILE helper: model classes only
READS     nothing
WRITES    nothing
USED BY   train.py
RUN       not run directly
------------------------------------------------------------------------------

Both models take a batch of standardized images (N, 1, 32, 32) and return
(N, K) logits (no softmax: nn.CrossEntropyLoss applies it).

  ANN (baseline)  flatten -> FC 512 -> FC 256 -> K
  CNN (primary)   3 x [conv 3x3 -> batch norm -> ReLU -> max pool 2x2] -> FC 256 -> K
"""
import torch.nn as nn
import torch.nn.functional as F

IMG = 32 # input images are 1 x 32 x 32


def out_size(n, k, p, s):
    """Side length after a conv / pool layer: input side n, kernel k, padding p, stride s."""
    return (n + 2 * p - k) // s + 1


class ANN(nn.Module):
    """Baseline: multilayer perceptron on the flattened image. It has no spatial
    structure, so the CNN - ANN gap shows what convolution adds."""
    def __init__(self, num_classes, hidden1=512, hidden2=256, dropout=0.3):
        super(ANN, self).__init__()
        self.name = "ann"
        self.hidden_units = hidden1

        self.N = IMG * IMG
        self.fc1 = nn.Linear(self.N, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = x.view(-1, self.N)
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class CNN(nn.Module):
    """Primary model: three conv blocks (C1, C2, C3 channels; side 32 -> 16 -> 8 -> 4),
    then one hidden fully connected layer. Batch norm keeps training stable; 
    padding kernel // 2 keeps the side length through each convolution."""
    def __init__(self, num_classes, C1=32, C2=64, C3=128, hidden_units=256, kernel=3,
                pool=2, dropout=0.3):
        super(CNN, self).__init__()
        self.name = "cnn"
        self.hidden_units = hidden_units
        self.kernel = kernel
        p = kernel // 2 # "same" padding

        self.conv1 = nn.Conv2d(1, C1, kernel, padding=p)
        self.bn1 = nn.BatchNorm2d(C1)
        self.conv2 = nn.Conv2d(C1, C2, kernel, padding=p)
        self.bn2 = nn.BatchNorm2d(C2)
        self.conv3 = nn.Conv2d(C2, C3, kernel, padding=p)
        self.bn3 = nn.BatchNorm2d(C3)
        self.pool = nn.MaxPool2d(pool, pool)

        n = IMG
        for _ in range(3): # conv (same size) then pool (halves)
            n = out_size(out_size(n, kernel, p, 1), pool, 0, pool)
        self.N = C3 * n * n # 128 * 4 * 4 = 2048

        self.fc1 = nn.Linear(self.N, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, self.N) # flatten
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x) # K logits
        return x


MODELS = {"ann": ANN, "cnn": CNN}
