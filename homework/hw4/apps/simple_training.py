import sys
sys.path.append('../python')
import needle as ndl
import needle.nn as nn
from needle import backend_ndarray as nd
from models import *
import time

device = ndl.cuda()

### CIFAR-10 training ###

def epoch_general_cifar10(dataloader, model, loss_fn=nn.SoftmaxLoss(), opt=None):
    """
    Iterates over the dataloader. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    correct, total_loss = 0, 0

    if opt is None:
        model.eval()
        for batch in dataloader:
            X, y = batch
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            loss = loss_fn(out, y)
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            total_loss += loss.data.numpy() * y.shape[0]
    else:
        model.train()
        for batch in dataloader:
            opt.reset_grad()
            X, y = batch
            X, y = ndl.Tensor(X, device=device), ndl.Tensor(y, device=device)
            out = model(X)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            correct += np.sum(np.argmax(out.numpy(), axis=1) == y.numpy())
            total_loss += loss.data.numpy() * y.shape[0]

    sample_nums = len(dataloader.dataset)
    return correct / sample_nums, total_loss / sample_nums
    ### END YOUR SOLUTION


def train_cifar10(model, dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
          lr=0.001, weight_decay=0.001, loss_fn=nn.SoftmaxLoss()):
    """
    Performs {n_epochs} epochs of training.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(n_epochs):
        avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn, opt=opt)
        print(f"Epoch: {epoch}, Acc: {avg_acc}, Loss: {avg_loss}")
    ### END YOUR SOLUTION


def evaluate_cifar10(model, dataloader, loss_fn=nn.SoftmaxLoss()):
    """
    Computes the test accuracy and loss of the model.

    Args:
        dataloader: Dataloader instance
        model: nn.Module instance
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_cifar10(dataloader, model, loss_fn=loss_fn)
    print(f"Evaluation Acc: {avg_acc}, Evaluation Loss: {avg_loss}")
    ### END YOUR SOLUTION


### PTB training ###
def epoch_general_ptb(data, model, seq_len=40, loss_fn=nn.SoftmaxLoss(), opt=None,
        clip=None, device=None, dtype="float32"):
    """
    Iterates over the data. If optimizer is not None, sets the
    model to train mode, and for each batch updates the model parameters.
    If optimizer is None, sets the model to eval mode, and simply computes
    the loss/accuracy.

    Args:
        data: data of shape (nbatch, batch_size) given from batchify function
        model: LanguageModel instance
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module instance
        opt: Optimizer instance (optional)
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    losses = []
    corrects = []
    dataset_size = 0
    train = opt is not None
    if train:
        model.train()
    else:
        model.eval()
    
    nbatch, batch_size = data.shape
    
    hidden = None
    for i in range(0, nbatch - 1, seq_len):
        x, y = ndl.data.get_batch(data, i, seq_len, device=device, dtype=dtype)

        batch_size = y.shape[0]
        dataset_size += batch_size
        y_pred, hidden = model(x, hidden)
        # detach the hidden state to avoid blowing up computational graph
        # between training on different sequences
        if isinstance(hidden, tuple):
            h, c = hidden
            hidden = (h.detach(), c.detach())
        else:
            hidden = hidden.detach()
        
        loss = loss_fn(y_pred, y)
        
        if train:
            opt.reset_grad()
            loss.backward()
            opt.step()

        losses.append(loss.numpy() * batch_size)
        correct = np.sum(y_pred.numpy().argmax(axis = 1) == y.numpy())
        corrects.append(correct)

    avg_acc = np.sum(np.array(corrects)) / dataset_size
    avg_loss = np.sum(np.array(losses)) / dataset_size
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def train_ptb(model, data, seq_len=40, n_epochs=1, optimizer=ndl.optim.SGD,
          lr=4.0, weight_decay=0.0, loss_fn=nn.SoftmaxLoss(), clip=None,
          device=None, dtype="float32"):
    """
    Performs {n_epochs} epochs of training.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        n_epochs: number of epochs (int)
        optimizer: Optimizer class
        lr: learning rate (float)
        weight_decay: weight decay (float)
        loss_fn: nn.Module class
        clip: max norm of gradients (optional)

    Returns:
        avg_acc: average accuracy over dataset from last epoch of training
        avg_loss: average loss over dataset from last epoch of training
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    opt = optimizer(model.parameters(), lr = lr, weight_decay = weight_decay)

    for _ in range(n_epochs):
        avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=opt, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


def evaluate_ptb(model, data, seq_len=40, loss_fn=nn.SoftmaxLoss(),
        device=None, dtype="float32"):
    """
    Computes the test accuracy and loss of the model.

    Args:
        model: LanguageModel instance
        data: data of shape (nbatch, batch_size) given from batchify function
        seq_len: i.e. bptt, sequence length
        loss_fn: nn.Module class

    Returns:
        avg_acc: average accuracy over dataset
        avg_loss: average loss over dataset
    """
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    avg_acc, avg_loss = epoch_general_ptb(data, model, seq_len=seq_len, loss_fn=loss_fn, opt=None, device=device, dtype=dtype)
    return avg_acc, avg_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    ### For testing purposes
    device = ndl.cpu()
    #dataset = ndl.data.CIFAR10Dataset("./data/cifar-10-batches-py", train=True)
    #dataloader = ndl.data.DataLoader(\
    #         dataset=dataset,
    #         batch_size=128,
    #         shuffle=True
    #         )
    #
    #model = ResNet9(device=device, dtype="float32")
    #train_cifar10(model, dataloader, n_epochs=10, optimizer=ndl.optim.Adam,
    #      lr=0.001, weight_decay=0.001)

    corpus = ndl.data.Corpus("./data/ptb")
    seq_len = 40
    batch_size = 16
    hidden_size = 100
    train_data = ndl.data.batchify(corpus.train, batch_size, device=device, dtype="float32")
    model = LanguageModel(1, len(corpus.dictionary), hidden_size, num_layers=2, device=device)
    train_ptb(model, train_data, seq_len, n_epochs=10, device=device)
