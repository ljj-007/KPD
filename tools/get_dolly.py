import datasets
import os
import re

dataset = datasets.load_dataset('databricks/databricks-dolly-15k', split='train')

os.mkdir("data/dolly", exist_ok=True)

num = 0
with open("data/dolly/data.txt", "w", encoding="utf-8") as wf:
    for data in dataset:
        wf.write(re.sub(r"\n+", "<@x(x!>", data['text']) + "\n")
        num += 1

print("Number of lines:", num)