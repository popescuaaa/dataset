from electric_dataset import ElectricProductionDataset, features
from torch.utils.data import DataLoader
import torch 

if __name__ == '__main__':
	csv_path = './csv/Electric_Production.csv'
	seq_len = 8
	batch_size = 10

	ds = ElectricProductionDataset(csv_path, seq_len)
	dl = DataLoader(ds, batch_size, shuffle = True, num_workers = 10)

	for idx, data in enumerate(dl):
		print(data.shape)
		assert data.shape == torch.Size([batch_size, seq_len]), 'Faild to produce correct samples size'