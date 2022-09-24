import torch
torch._C._jit_set_nvfuser_enabled(False)

from transformers import BertForSequenceClassification
import torchdynamo
from torchdynamo.optimizations.training import aot_ort, aot_eager, aot_mem_efficient_fusion, aot_cudagraphs
from onnxruntime.training.ortmodule import ORTModule

def parse_args():
  import argparse
  parser = argparse.ArgumentParser(description='')
  parser.add_argument('--mode', type=str, default='ort', choices=['ort', 'eager', 'aten', 'nvfuser', 'cudagraphs', 'ortmodule'])
  return parser.parse_args()

args = parse_args()


device = 'cuda'
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', return_dict=True).to(device)
model.train()

from transformers import AdamW
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
text_batch = ["I love Pixar.", "I don't care for Pixar."]
encoding = tokenizer(text_batch, return_tensors='pt', padding=True, truncation=True)
input_ids = encoding['input_ids'].to(device)
attention_mask = encoding['attention_mask'].to(device)

if args.mode == 'aten':
  modle = torchdynamo.optimize(aot_eager)(model)
elif args.mode == 'ort':
  model = torchdynamo.optimize(aot_ort)(model)
elif args.mode == 'eager':
  pass
elif args.mode == 'nvfuser':
  model = torchdynamo.optimize(aot_mem_efficient_fusion)(model)
elif args.mode == 'cudagraphs':
  model = torchdynamo.optimize(aot_cudagraphs)(model)
elif args.mode == 'ortmodule':
  model = ORTModule(model)
else:
  raise ValueError('Unknown mode {}'.format(args.mode))


no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)
labels = torch.tensor([1,0]).unsqueeze(0).to(device)
for _ in range(4):
  torch.cuda.nvtx.range_push(f'Iter{_}')

  torch.cuda.nvtx.range_push('F')
  outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
  torch.cuda.nvtx.range_pop()

  #optimizer.zero_grad()
  loss = outputs.loss

  torch.cuda.nvtx.range_push('B')
  loss.backward()
  print(loss)
  torch.cuda.nvtx.range_pop()

  #torch.cuda.nvtx.range_push('O')
  #optimizer.step()
  #torch.cuda.nvtx.range_pop()

  torch.cuda.nvtx.range_pop()