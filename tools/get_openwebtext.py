import datasets
import os
import re
from tqdm import tqdm

# dataset = datasets.load_dataset('openwebtext', split='train')

# os.makedirs("data/openwebtext", exist_ok=True)

dataset = datasets.load_dataset("/data2/jun/my-distill/data/openwebtext", split='train')

num = 0
with open("data/openwebtext/data.txt", "w") as f:
    for data in tqdm(dataset):
        f.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num) # 8,092,112