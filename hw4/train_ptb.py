import needle as ndl
sys.path.append('./apps')
from models import LanguageModel
from simple_training import train_ptb, evaluate_ptb

device = ndl.cuda()
corpus = ndl.data.Corpus("data/ptb")
train_data = ndl.data.batchify(corpus.train, batch_size=16, device=device, dtype="float32")
model = LanguageModel(30, len(corpus.dictionary), hidden_size=10, num_layers=2, seq_model='lstm', device=device)
train_ptb(model, train_data, seq_len=1, n_epochs=1, device=device)
evaluate_ptb(model, train_data, seq_len=40, device=device)