import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from src.model.config import data_config



class CustomDataset(Dataset):
    def __init__(self,path,block_size):
        super().__init__()
        self.path = path
        self.block_size = block_size

        # now as every shard_file has different number of tokens, this creates a risk of index error
        self.blocks_per_shard = []

        for p in self.path:
            file = np.load(p,mmap_mode='r')
            num_blocks = (len(file)-1)//block_size
            self.blocks_per_shard.append(num_blocks)
        self.cumsum = np.cumsum(self.blocks_per_shard)

    def __len__(self):
        return int(self.cumsum[-1])

    
    def __getitem__(self,index):
        idx = 0
        for i in range(len(self.cumsum)):
            if self.cumsum[i] > index:
                idx = i
                break
        
        local_idx = 0
        if idx ==0 :
            local_idx = index
        else:
            local_idx = index - int(self.cumsum[idx-1])

        shard = np.load(self.path[idx] , mmap_mode='r')# this mmap_mode = 'r' do not loads the whole file into ram, just the sliced array
        start = local_idx*self.block_size

        x = shard[start : start + self.block_size]
        y = shard[start+1 : start+self.block_size+1]

        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))

    
class get_dataloader:
    def __init__(self, data_config, train_path, val_path):
        super().__init__()
        assert train_path is not None, "train_path cannot be None"
        assert val_path is not None, "val_path cannot be None"
        self.train = CustomDataset(train_path)
        self.val = CustomDataset(val_path)

        assert data_config is not None, "data_config cannot be None"
        self.data_config = data_config

        
    def train_dataloader(self):
        dataset = CustomDataset(self.train_path,self.block_size)
        return DataLoader(
            self.train,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor,
            pin_memory=self.data_config.pin_memory,
            in_order=self.data_config.in_order,
            shuffle = True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.data_config.batch_size,
            num_workers=self.data_config.num_workers,
            prefetch_factor=self.data_config.prefetch_factor,
            pin_memory=self.data_config.pin_memory,
            in_order=self.data_config.in_order,
            shuffle = False
        )



# for testing purpose only
if __name__ =="__main__":
    train_path = ['corpus/shard_0.npy']
    val_path = ['corpus/shard_1.npy']

    ds = get_dataloader(train_path,val_path,block_size=512,batch_size=8,num_workers=2)

    train_loader = ds.train_dataloader()

    for i in range(1):
        sample_x,sample_y = next(iter(train_loader))

        print(sample_x.shape)
        print(sample_y.shape)