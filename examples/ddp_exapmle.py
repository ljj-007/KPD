import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

def main():
    """ 几个gpu, 开几个主函数的进程 """
    # print(dist.is_initialized()) # False
    dist.init_process_group(backend='nccl', init_method='env://')
    # print(dist.is_initialized()) # True
    local_rank = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(local_rank)

    # 模型、数据集和数据加载器
    model = nn.Linear(10, 10).cuda()
    ddp_model = DDP(model, device_ids=[local_rank])
    # print(ddp_model.device)

    dataset = TensorDataset(torch.randn(100, 10), torch.randn(100, 10))
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(dataset, batch_size=32, sampler=train_sampler)

    criterion = nn.MSELoss().cuda()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    # 训练循环
    for epoch in range(10):
        train_sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
