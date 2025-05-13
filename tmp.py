import datasets
import os
import re
from tqdm import tqdm
import torch
import torch.nn as nn


cri = nn.CrossEntropyLoss()
print(cri.ignore_index)
# logits = torch.randn(3, 5, requires_grad=True)
# targets = torch.tensor([1, -1, 4])
# print(cri(logits, targets))



# with open("/data2/jun/my-distill/data/openwebtext/data.txt", "r", encoding="utf-8") as rf:
#     lines = rf.readlines()
#     lines = lines[:2]
#     for line in lines:
#         print(line)


# dataset = datasets.load_dataset("/data2/jun/my-distill/data/openwebtext", split='train')

# dataset = dataset[:5]
# for i in range(5):
#     print(dataset["text"][i])

# num = 0
# with open("data/openwebtext/data.txt", "w") as f:
#     for data in tqdm(dataset):
#         f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
#         num += 1

# print("Number of lines:", num) # 8,092,112