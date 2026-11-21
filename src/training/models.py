"""models.py: the neural networks (same template as Lab 3).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: model classes and their default sizes
READS     nothing
WRITES    nothing
USED BY   training/tune.py, training/train.py, evaluation/final_test.py
RUN       not run directly
------------------------------------------------------------------------------

Both models take a batch of standardized images (N, 1, 32, 32) and return
(N, K) logits (no softmax: nn.CrossEntropyLoss applies it).

  ANN (baseline)  flatten -> FC hidden1 -> FC hidden2 -> K
  CNN (primary)   num_layers x [conv 3x3 -> batch norm -> ReLU -> max pool 2x2] -> FC hidden_units -> K
                  channels c, 2c, 4c, ... (default 3 layers)

Sizes were reduced to cut training time (32x32 inputs do not need more):
  ANN  512-256 -> 256-128        697k -> 316k parameters
  CNN  32-64-128, FC 256 -> 16-32-64, FC 128   659k -> 175k parameters, ~4x fewer FLOPs
tune.py searches around these defaults; train.py uses the tuned values if they exist.
"""
import torch.nn as nn
import torch.nn.functional as F

IMG = 32 # input images are 1 x 32 x 32


def out_size(n, k, p, s):
    """Side length after a conv / pool layer: input side n, kernel k, padding p, stride s."""
    return (n + 2 * p - k) // s + 1


class ANN(nn.Module):
    """Baseline: multilayer perceptron on the flattened image."""
    def __init__(self, num_classes, hidden1=256, hidden2=128, dropout=0.3):
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
    """Primary model: num_layers conv blocks (c, 2c, 4c, ... channels; each pool halves
    the side, e.g. 32 -> 16 -> 8 -> 4 for 3 layers), then one hidden fully connected layer."""
    def __init__(self, num_classes, channels=16, hidden_units=128, num_layers=3, kernel=3, pool=2,
                 dropout=0.3):
        super(CNN, self).__init__()
        self.name = "cnn"
        self.hidden_units = hidden_units
        self.kernel = kernel
        self.num_layers = num_layers
        p = kernel // 2 # "same" padding

        # conv blocks conv1/bn1, conv2/bn2, ... (names kept so older checkpoints still load)
        c_in = 1
        for i in range(1, num_layers + 1):
            c_out = channels * 2 ** (i - 1)
            setattr(self, f"conv{i}", nn.Conv2d(c_in, c_out, kernel, padding=p))
            setattr(self, f"bn{i}", nn.BatchNorm2d(c_out))
            c_in = c_out
        self.pool = nn.MaxPool2d(pool, pool)

        # side length after all blocks: conv keeps the size, pool halves it
        n = IMG
        for _ in range(num_layers):
            n = out_size(out_size(n, kernel, p, 1), pool, 0, pool)
        assert n >= 1, f"{num_layers} conv blocks shrink a {IMG}x{IMG} image to nothing"
        self.N = c_in * n * n

        self.fc1 = nn.Linear(self.N, hidden_units)
        self.fc2 = nn.Linear(hidden_units, num_classes)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # conv -> batch norm -> ReLU -> max pool, once per block
        for i in range(1, self.num_layers + 1):
            conv, bn = getattr(self, f"conv{i}"), getattr(self, f"bn{i}")
            x = self.pool(F.relu(bn(conv(x))))
        x = x.view(-1, self.N) # flatten
        x = self.drop(F.relu(self.fc1(x)))
        x = self.fc2(x) # K logits
        return x


MODELS = {"ann": ANN, "cnn": CNN}
DEFAULTS = {"ann": dict(hidden1=256, hidden2=128), # size hyperparameters per model
            "cnn": dict(channels=16, hidden_units=128, num_layers=3)}


def build_model(name, num_classes, model_config=None):
    """MODELS[name] with DEFAULTS[name] overridden by model_config (e.g. from tuning)."""
    cfg = dict(DEFAULTS[name], **(model_config or {}))
    return MODELS[name](num_classes, **cfg), cfg


def count_params(net):
    return sum(p.numel() for p in net.parameters())
