"""
No data augmentation:
Epoch: 0, Acc: 0.38678, Loss: [1.7120664]
Epoch: 1, Acc: 0.49422, Loss: [1.4014741]
Epoch: 2, Acc: 0.54014, Loss: [1.2785015]
Epoch: 3, Acc: 0.57622, Loss: [1.1893052]
Epoch: 4, Acc: 0.60468, Loss: [1.1160048]
Epoch: 5, Acc: 0.6258, Loss: [1.0537735]
Epoch: 6, Acc: 0.64792, Loss: [0.9968544]
Epoch: 7, Acc: 0.66454, Loss: [0.9435891]
Epoch: 8, Acc: 0.68322, Loss: [0.89590883]
Epoch: 9, Acc: 0.69688, Loss: [0.8593632]
Evaluation Acc: 0.66702, Evaluation Loss: [0.9336795]
"""

import sys
sys.path.append('./python')
sys.path.append('./apps')

import needle as ndl
from models import ResNet9
from simple_training import train_cifar10, evaluate_cifar10

device = ndl.cuda()
train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
train_dataloader = ndl.data.DataLoader(dataset=train_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       device=device)
test_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=False)
test_dataloader = ndl.data.DataLoader(dataset=test_dataset,
                                       batch_size=128,
                                       shuffle=True,
                                       device=device)

model = ResNet9(device=device, dtype="float32")
best_acc = -1
for _ in range(20):
      train_cifar10(model, train_dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                    lr=0.0005, weight_decay=0.001)
      evaluate_cifar10(model, test_dataloader)
for _ in range(20):
      train_cifar10(model, train_dataloader, n_epochs=1, optimizer=ndl.optim.Adam,
                    lr=0.0001, weight_decay=0.001)
      evaluate_cifar10(model, test_dataloader)