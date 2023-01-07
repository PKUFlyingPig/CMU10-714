import sys
sys.path.append("src/")

# Reload the simple_ml module to include the newly-compiled C++ extension
import importlib
import simple_ml
importlib.reload(simple_ml)

from simple_ml import train_softmax, parse_mnist

X_tr, y_tr = parse_mnist("data/train-images-idx3-ubyte.gz", 
                         "data/train-labels-idx1-ubyte.gz")
X_te, y_te = parse_mnist("data/t10k-images-idx3-ubyte.gz",
                         "data/t10k-labels-idx1-ubyte.gz")

train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr = 0.2, batch=100, cpp=True)