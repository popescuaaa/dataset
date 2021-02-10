import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from torch.utils.data import Dataset, DataLoader

csv_path = './csv/Electric_Production.csv'

class ElectricProductionDataset(Dataset):
	def __init__(self, file_path: str, seq_len: int):
		self.df = pd.read_csv(file_path)

		# Claculate ∆t (deltas)
		self.dt = np.array([ (self.df.Value[i + 1] - self.df.Value[i]) for i in range(self.df.Value.size - 1)])
		self.dt = np.concatenate([np.array([0]), self.dt])
		
		# Append ∆t (deltas)
		self.df.insert(2, 'dt', self.dt)
		
		# Create an array of sequences
		self.data = [self.df[i:i+seq_len] for i in range(self.df.Value - seq_len)]

	def __len__(self):
		pass

	def __getitem__(self):
		pass

	def sample(self):
		pass

if __name__ == '__main__':
	ds = ElectricProductionDataset(csv_path, 10)