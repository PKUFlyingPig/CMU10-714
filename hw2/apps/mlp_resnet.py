import sys

from needle.data import MNISTDataset, DataLoader
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
import numpy as np
import time
import os

np.random.seed(0)

def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    main_path = nn.Sequential(nn.Linear(dim, hidden_dim), norm(hidden_dim), nn.ReLU(), 
                              nn.Dropout(drop_prob), nn.Linear(hidden_dim, dim), norm(dim))
    res = nn.Residual(main_path)
    return nn.Sequential(res, nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(dim, hidden_dim=100, num_blocks=3, num_classes=10, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    resnet = nn.Sequential(nn.Linear(dim, hidden_dim), nn.ReLU(), 
                           *[ResidualBlock(dim=hidden_dim, hidden_dim=hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)],
                           nn.Linear(hidden_dim, num_classes))
    return resnet
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    if opt is None:    
        model.eval()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
    else:
        model.train()
        for X, y in dataloader:
            logits = model(X)
            loss = loss_fn(logits, y)
            tot_error += np.sum(logits.numpy().argmax(axis=1) != y.numpy())
            tot_loss.append(loss.numpy())
            opt.reset_grad()
            loss.backward()
            opt.step()
    sample_nums = len(dataloader.dataset)
    return tot_error/sample_nums, np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(batch_size=100, epochs=10, optimizer=ndl.optim.Adam,
                lr=0.001, weight_decay=0.001, hidden_dim=100, data_dir="data"):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_loader, resnet, opt)
    test_err, test_loss = epoch(test_loader, resnet, None)
    return train_err, train_loss, test_err, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    np.random.seed(1)
    train_mnist(250, 5, ndl.optim.SGD, 0.01, 0.001, 100, data_dir="../data")
