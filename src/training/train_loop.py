"""train_loop.py: the training loop (same template as Lab 3).

PIPELINE  data_collection/: 1 get_fonts -> 2 make_dataset -> 3 data_figures -> 4 hasy
          training/:        5 tune (optional) -> 6 train
          evaluation/:      7 final_test (once, at the end)
THIS FILE helper: set_seed, get_model_name, evaluate, train_net
READS     datasets from data.py
WRITES    <results>/ckpt/<model name>      one checkpoint per epoch
          <results>/curves/<run>_{train_err,train_loss,val_err,val_loss}.csv
          (<results> = results/ for train.py, results/tuning/ for tune.py)
USED BY   training/tune.py, training/train.py
RUN       not run directly
------------------------------------------------------------------------------

Differences from the Lab 3 template (kept from the earlier runs of this project):
  * optimizer Adam with weight decay and a cosine learning-rate schedule
    (instead of SGD with momentum)
  * checkpoints and curves go into results/ instead of the working directory
"""
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from data import get_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=360):
    """Seed Python, numpy and PyTorch so a run is repeatable. Seeding covers weight init,
    dropout, shuffling and the weighted sampler; the cuDNN flags make GPU convolutions
    deterministic too (same GPU + library versions -> identical results; slightly slower)."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_model_name(name, batch_size, learning_rate, epoch):
    """ Generate a name for the model consisting of all the hyperparameter values

    Args:
        name: model name, e.g. "cnn" or "cnn_ft" (fine-tuned)
    Returns:
        path: A string with the hyperparameter name and value concatenated
    """
    path = "{0}_bs{1}_lr{2}_epoch{3}".format(name, batch_size, learning_rate, epoch)
    return path


def evaluate(net, loader, criterion):
    """ Evaluate the network on a data set (e.g. the validation set).

     Args:
         net: PyTorch neural network object
         loader: PyTorch data loader for the data set
         criterion: The loss function
     Returns:
         err: A scalar for the avg classification error over the data set
         loss: A scalar for the average loss function over the data set
     """
    net.eval()
    total_loss = 0.0
    total_err = 0.0
    total_epoch = 0
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            wrong = outputs.argmax(dim=1) != labels
            total_err += int(wrong.sum())
            total_loss += loss.item()
            total_epoch += len(labels)
    err = float(total_err) / total_epoch
    loss = float(total_loss) / (i + 1)
    return err, loss


def train_net(net, train_set, val_set, batch_size=256, learning_rate=1e-3, num_epochs=12,
              weight_decay=1e-4, tag="", results="results", sample_weights=None):
    """ Train `net`, checkpoint every epoch, and keep track of the best epoch.

    Args:
        net: the model (an nn.Module from models.py); it is trained in place
        train_set, val_set: PyTorch datasets; model selection uses val_set
        tag: added to the model name, e.g. "_ft" for fine-tuning
        sample_weights: optional weight per training image (class-balanced sampling)
    Returns:
        best_path: checkpoint of the epoch with the lowest validation error
        best_val_err: that validation error
        curve_path: prefix of the 4 CSV files with the training curves
        elapsed_time: training time in seconds
    """
    net.to(device)
    ckpt_dir = os.path.join(results, "ckpt"); os.makedirs(ckpt_dir, exist_ok=True)
    curve_dir = os.path.join(results, "curves"); os.makedirs(curve_dir, exist_ok=True)
    ########################################################################
    # Obtain the PyTorch data loader objects to load batches of the datasets
    train_loader = get_loader(train_set, batch_size, weights=sample_weights)
    val_loader = get_loader(val_set, batch_size, shuffle=False)
    ########################################################################
    # Define the Loss function and optimizer
    # The loss function is Cross Entropy Loss: it takes the unnormalized
    # output (logits) of the network and the integer class label.
    # The optimizer is Adam with weight decay; the learning rate follows a
    # cosine schedule from learning_rate down to 0 over num_epochs.
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    ########################################################################
    # Set up some numpy arrays to store the training/val loss and error
    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)
    ########################################################################
    # Train the network
    # Loop over the data iterator and sample a new batch of training data
    # Get the output from the network, and optimize our loss function.
    start_time = time.time()

    best_val_err = np.inf   # best validation error
    best_path = None
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        net.train()
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            # Get the inputs
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # Calculate the statistics
            wrong = outputs.argmax(dim=1) != labels
            total_train_err += int(wrong.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)
        scheduler.step()
        train_err[epoch] = total_train_err / total_epoch
        train_loss[epoch] = total_train_loss / (i + 1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {:.4f}, Train loss: {:.4f} | " +
               "Validation err: {:.4f}, Validation loss: {:.4f} ({:.0f}s)").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch],
                   time.time() - start_time))
        # Save the current model (checkpoint) to a file
        model_path = os.path.join(ckpt_dir, get_model_name(net.name + tag, batch_size,
                                                           learning_rate, epoch))
        torch.save(net.state_dict(), model_path)
        # Save the best model
        if val_err[epoch] < best_val_err:
            best_val_err = val_err[epoch]
            best_path = model_path
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))
    # Write the train/val loss/err into CSV files for plotting later
    curve_path = os.path.join(curve_dir, get_model_name(net.name + tag, batch_size,
                                                        learning_rate, num_epochs - 1))
    np.savetxt("{}_train_err.csv".format(curve_path), train_err)
    np.savetxt("{}_train_loss.csv".format(curve_path), train_loss)
    np.savetxt("{}_val_err.csv".format(curve_path), val_err)
    np.savetxt("{}_val_loss.csv".format(curve_path), val_loss)

    return best_path, best_val_err, curve_path, elapsed_time


def load_curves(curve_path):
    """The 4 training curves saved by train_net, as a dict of lists."""
    return {k: np.loadtxt("{}_{}.csv".format(curve_path, k), ndmin=1).tolist()
            for k in ("train_err", "train_loss", "val_err", "val_loss")}
