import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import MinMaxScaler

class EnergyDataset(Dataset):
    def __init__(self, path: str, seq_len: int):
        self.original_data = np.loadtxt(path, delimiter = ",",skiprows = 1)
        self.seq_len = seq_len

        # Scale data
        self.scaler = MinMaxScaler()
        self.original_data = self.scaler(torch.from_numpy(self.original_data)).numpy()
        
        # Cut data in sequences
        _data = []
        for i in range(0, len(self.original_data) - self.seq_len):
            _x = self.original_data[i:i + seq_len]
            _data.append(_x)

        self.data = [torch.from_numpy(np.array(_data[i])) for i in range(len(_data))]
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


if __name__ == '__main__':
    """
    Test energy dataset
    """

    path = './csv/energy.csv'
    ds = EnergyDataset(path, 100)
    dl = DataLoader(ds, batch_size = 10, shuffle = False)
    for idx, data in enumerate(dl):
        print(data)
        break