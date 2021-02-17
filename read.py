import torch

p = 'results.pt.tar'
print(torch.__version__)
result = torch.load(p).get('result')

print()
