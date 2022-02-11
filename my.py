from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lazy_tensor_core
from onnxruntime.capi import _pybind_state as ost
import lazy_tensor_core.core.lazy_model as ltm
torch.manual_seed(0)

ost.register_ort_as_torch_jit_executor()
lazy_tensor_core._LAZYC._ltc_init_ts_backend()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, bias=False)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128, bias=False)
        self.fc2 = nn.Linear(128, 10, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

device = 'lazy'
x = torch.rand((64, 1, 28, 28), dtype=torch.float32).to(device)
y = torch.randint(0, 9, (64,), dtype=torch.int64).to(device)
model = Net().to(device)
optimizer = optim.Adagrad(model.parameters(), lr=0.001)
for i in range(5):
  print(i)
  optimizer.zero_grad()
  print(f'tr::forward::{i}')
  output = model(x)
  ltm.mark_step()
  print('tr:loss')
  loss = F.nll_loss(output, y)
  print(loss)
  print(f'tr::backward::{i}')
  loss.backward() #2.1838
  ltm.mark_step()
  print(f'tr::optim::{i}')
  optimizer.step()
  ltm.mark_step()