import sys
sys.path.append('./python')
sys.path.append('./apps')
import numpy as np
import pytest
import torch
import itertools
import mugrade

import needle as ndl
import needle.nn as nn

from simple_training import *
from models import LanguageModel


np.random.seed(3)


_DEVICES = [ndl.cpu(), pytest.param(ndl.cuda(),
    marks=pytest.mark.skipif(not ndl.cuda().enabled(), reason="No GPU"))]


BATCH_SIZES = [1, 15]
INPUT_SIZES = [1, 11]
HIDDEN_SIZES = [1, 12]
BIAS = [True, False]
INIT_HIDDEN = [True, False]
NONLINEARITIES = ['tanh', 'relu']
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn_cell(batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNNCell(input_size, hidden_size, nonlinearity=nonlinearity, bias=bias)
    if init_hidden:
        h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        h_ = model_(torch.tensor(x), None)

    model = nn.RNNCell(input_size, hidden_size, device=device, bias=bias, nonlinearity=nonlinearity)
    model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)
    if init_hidden:
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        h = model(ndl.Tensor(x, device=device), None)
    assert h.device == device
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm_cell(batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTMCell(input_size, hidden_size, bias=bias)
    if init_hidden:
        h_, c_ = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        h_, c_ = model_(torch.tensor(x), None)

    model = nn.LSTMCell(input_size, hidden_size, device=device, bias=bias)

    model.W_ih = ndl.Tensor(model_.weight_ih.detach().numpy().transpose(), device=device)
    model.W_hh = ndl.Tensor(model_.weight_hh.detach().numpy().transpose(), device=device)
    if bias:
        model.bias_ih = ndl.Tensor(model_.bias_ih.detach().numpy(), device=device)
        model.bias_hh = ndl.Tensor(model_.bias_hh.detach().numpy(), device=device)

    if init_hidden:
        h, c = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        h, c = model(ndl.Tensor(x, device=device), None)
    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)

    h.sum().backward()
    h_.sum().backward()
    np.testing.assert_allclose(model_.weight_ih.grad.detach().numpy().transpose(), model.W_ih.grad.numpy(), atol=1e-5, rtol=1e-5)


SEQ_LENGTHS = [1, 13]
NUM_LAYERS = [1, 2]
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("nonlinearity", NONLINEARITIES)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_rnn(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, nonlinearity, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.RNN(input_size, hidden_size, num_layers=num_layers, bias=bias, nonlinearity=nonlinearity)
    if init_hidden:
        output_, h_ = model_(torch.tensor(x), torch.tensor(h0))
    else:
        output_, h_ = model_(torch.tensor(x), None)

    model = nn.RNN(input_size, hidden_size, num_layers, bias, device=device, nonlinearity=nonlinearity)
    for k in range(num_layers):
        model.rnn_cells[k].W_ih = ndl.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.rnn_cells[k].W_hh = ndl.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.rnn_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.rnn_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
    else:
        output, h = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.rnn_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("input_size", INPUT_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("bias", BIAS)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_lstm(seq_length, num_layers, batch_size, input_size, hidden_size, bias, init_hidden, device):
    x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
    h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
    c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)

    model_ = torch.nn.LSTM(input_size, hidden_size, bias=bias, num_layers=num_layers)
    if init_hidden:
        output_, (h_, c_) = model_(torch.tensor(x), (torch.tensor(h0), torch.tensor(c0)))
    else:
        output_, (h_, c_) = model_(torch.tensor(x), None)

    model = nn.LSTM(input_size, hidden_size, num_layers, bias, device=device)
    for k in range(num_layers):
        model.lstm_cells[k].W_ih = ndl.Tensor(getattr(model_, f'weight_ih_l{k}').detach().numpy().transpose(), device=device)
        model.lstm_cells[k].W_hh = ndl.Tensor(getattr(model_, f'weight_hh_l{k}').detach().numpy().transpose(), device=device)
        if bias:
            model.lstm_cells[k].bias_ih = ndl.Tensor(getattr(model_, f'bias_ih_l{k}').detach().numpy(), device=device)
            model.lstm_cells[k].bias_hh = ndl.Tensor(getattr(model_, f'bias_hh_l{k}').detach().numpy(), device=device)
    if init_hidden:
        output, (h, c) = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
    else:
        output, (h, c) = model(ndl.Tensor(x, device=device), None)

    np.testing.assert_allclose(h_.detach().numpy(), h.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(c_.detach().numpy(), c.numpy(), atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(output_.detach().numpy(), output.numpy(), atol=1e-5, rtol=1e-5)

    output.sum().backward()
    output_.sum().backward()
    np.testing.assert_allclose(model.lstm_cells[0].W_ih.grad.detach().numpy(), model_.weight_ih_l0.grad.numpy().transpose(), atol=1e-5, rtol=1e-5)


OUTPUT_SIZES = [1, 1000]
EMBEDDING_SIZES = [1, 34]
SEQ_MODEL = ['rnn', 'lstm']
@pytest.mark.parametrize("seq_length", SEQ_LENGTHS)
@pytest.mark.parametrize("num_layers", NUM_LAYERS)
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("embedding_size", EMBEDDING_SIZES)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("init_hidden", INIT_HIDDEN)
@pytest.mark.parametrize("output_size", OUTPUT_SIZES)
@pytest.mark.parametrize("seq_model", SEQ_MODEL)
@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_language_model_implementation(seq_length, num_layers, batch_size, embedding_size, hidden_size,
                        init_hidden, output_size, seq_model, device):
    #TODO add test for just nn.embedding?
    x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
    h0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
    c0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)

    model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
    if init_hidden:
        if seq_model == 'lstm':
            h = (h0, c0)
        elif seq_model == 'rnn':
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)
    else:
        output, h_ = model(ndl.Tensor(x, device=device), None)

    if seq_model == 'lstm':
        assert isinstance(h_, tuple)
        h0_, c0_ = h_
        assert c0_.shape == (num_layers, batch_size, hidden_size)
    elif seq_model == 'rnn':
        h0_ = h_
    assert h0_.shape == (num_layers, batch_size, hidden_size)
    assert output.shape == (batch_size * seq_length, output_size)
    #TODO actually test values
    output.backward()
    for p in model.parameters():
        assert p.grad is not None

@pytest.mark.parametrize("device", _DEVICES, ids=["cpu", "cuda"])
def test_language_model_training(device):
    np.random.seed(0)
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 10
    num_examples = 100
    batch_size = 16
    seq_model = 'rnn'
    num_layers = 2
    hidden_size = 10
    n_epochs=2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
    model = LanguageModel(30, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers, seq_model=seq_model, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    if str(device) == "cpu()":
        np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)
    elif str(device) == "cuda()":
        np.testing.assert_allclose(5.711512, train_loss, atol=1e-5, rtol=1e-5)
        np.testing.assert_allclose(5.388685, test_loss, atol=1e-5, rtol=1e-5)




### MUGRADE ###

TEST_BATCH_SIZES = [6]
TEST_INPUT_SIZES = [3]
TEST_HIDDEN_SIZES = [5]
TEST_SEQ_LENGTHS = [7]
TEST_NUM_LAYERS = [3]
TEST_OUTPUT_SIZES = [16]
TEST_EMBEDDING_SIZES = [8]
TEST_SEQ_MODEL = ['rnn', 'lstm']

def mugrade_submit(x):
    if isinstance(x, np.ndarray):
        x = x.flatten()[:64]
        #print(x)
        mugrade.submit(x)
    else:
        #print(x)
        mugrade.submit(x)


def submit_rnn():
    #devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]

    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')

    for (device, batch_size, input_size, hidden_size) in itertools.product(
        devices, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        model = nn.RNNCell(input_size, hidden_size, device=device)
        mugrade_submit(model.W_ih.numpy())
        h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
        mugrade_submit(h.numpy())
        h.sum().backward()
        mugrade_submit(model.W_hh.grad.numpy())

    for (device, seq_length, num_layers, batch_size, input_size, hidden_size) in itertools.product(
        devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        model = nn.RNN(input_size, hidden_size, num_layers, device=device)
        output, h = model(ndl.Tensor(x, device=device), ndl.Tensor(h0, device=device))
        mugrade_submit(h.numpy())
        mugrade_submit(output.numpy())
        output.sum().backward()
        mugrade_submit(model.rnn_cells[-1].W_hh.grad.numpy())


def submit_lstm():
    #devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]
    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')
    for (device, batch_size, input_size, hidden_size) in itertools.product(
        devices, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        c0 = np.random.randn(batch_size, hidden_size).astype(np.float32)
        model = nn.LSTMCell(input_size, hidden_size, device=device)
        mugrade_submit(model.W_hh.numpy())
        (h, c) = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
        mugrade_submit(h.numpy())
        mugrade_submit(c.numpy())
        h.sum().backward()
        mugrade_submit(model.W_hh.grad.numpy())

    for (device, seq_length, num_layers, batch_size, input_size, hidden_size) in itertools.product(
        devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_INPUT_SIZES, TEST_HIDDEN_SIZES):
        x = np.random.randn(seq_length, batch_size, input_size).astype(np.float32)
        h0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        c0 = np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32)
        model = nn.LSTM(input_size, hidden_size, num_layers, device=device)
        output, (h, c) = model(ndl.Tensor(x, device=device), (ndl.Tensor(h0, device=device), ndl.Tensor(c0, device=device)))
        mugrade_submit(h.numpy())
        mugrade_submit(c.numpy())
        mugrade_submit(output.numpy())
        output.sum().backward()
        mugrade_submit(model.lstm_cells[-1].W_hh.grad.numpy())


def submit_language_model():
    #devices = [ndl.cpu(), ndl.cuda()] if ndl.cuda().enabled() else [ndl.cpu()]
    devices = [ndl.cpu(), ndl.cuda()]
    if not ndl.cuda().enabled():
        print('You need a GPU to run some of these tests.')
    for (device, seq_length, num_layers, batch_size, embedding_size, hidden_size, seq_model, output_size) in itertools.product(
        devices, TEST_SEQ_LENGTHS, TEST_NUM_LAYERS, TEST_BATCH_SIZES, TEST_EMBEDDING_SIZES, TEST_HIDDEN_SIZES, TEST_SEQ_MODEL, TEST_OUTPUT_SIZES):
        x = np.random.randint(0, output_size, (seq_length, batch_size)).astype(np.float32)
        h0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
        c0 = ndl.Tensor(np.random.randn(num_layers, batch_size, hidden_size).astype(np.float32), device=device)
        model = LanguageModel(embedding_size, output_size, hidden_size, num_layers, seq_model, device=device)
        if seq_model == 'lstm':
            h = (h0, c0)
        elif seq_model == 'rnn':
            h = h0
        output, h_ = model(ndl.Tensor(x, device=device), h)
        if seq_model == 'lstm':
            h0_, c0_ = h_
            mugrade_submit(c0_.numpy())
        elif seq_model == 'rnn':
            h0_ = h_
        mugrade_submit(h0_.numpy())
        mugrade_submit(output.numpy())

    device = ndl.cpu() # TODO CHANGE BACK
    # device = ndl.cpu()
    corpus = ndl.data.Corpus("data/ptb", max_lines=20)
    seq_len = 8
    num_examples = 88
    batch_size = 12
    seq_model = 'lstm'
    num_layers = 2
    hidden_size = 12
    n_epochs=2
    train_data = ndl.data.batchify(corpus.train, batch_size=batch_size, device=device, dtype="float32")
    model = LanguageModel(28, len(corpus.dictionary), hidden_size=hidden_size, num_layers=num_layers,
        seq_model=seq_model, device=device)
    train_acc, train_loss = train_ptb(model, train_data, seq_len=seq_len, n_epochs=n_epochs, device=device)
    test_acc, test_loss = evaluate_ptb(model, train_data, seq_len=seq_len, device=device)
    mugrade_submit(train_loss)
    mugrade_submit(test_loss)


if __name__ == "__main__":
    submit_rnn()
    submit_lstm()
    submit_language_model()
