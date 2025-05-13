import torch
import numpy as np
from data_utils.indexed_dataset import make_dataset, make_builder


""" Read Data """
dataset_path = "./processed_data/dolly/full/llama2/valid_0"
dataset_impl = "mmap"
dataset = make_dataset(dataset_path, dataset_impl)

""" Load Data """
index_file = "./tmp/tmp_dataset.idx" # meta information saved in this file
bin_file = "./tmp/tmp_dataset.bin" # data saved in this file
dtype = np.int32
builder = make_builder(bin_file, dataset_impl, dtype)
for i in range(10):
    tensor = torch.tensor([i, i+1, i+2])
    builder.add_item(tensor)
builder.end_document()
# Merge Data (option)
# another_file = 'path/to/another/dataset'
# builder.merge_file_(another_file)
builder.finalize(index_file)
print("Dataset has been saved")