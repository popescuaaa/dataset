import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as pyplot
from torch.utils.data import Dataset

features = { 1: 'Value', 2: 'dt' }

class ElectricProductionDataset(Dataset):
	def __init__(self, file_path: str, seq_len: int, feature: str or None = 'dt'):
		self.seq_len = seq_len
		self.feature = feature

		self.df = pd.read_csv(file_path)

		# Compute ∆t (deltas)
		self.dt = np.array([ (self.df.Value[i + 1] - self.df.Value[i]) 
			for i in range(self.df.Value.size - 1)])
		self.dt = np.concatenate([np.array([0]), self.dt])
		
		# Append ∆t (deltas)
		self.df.insert(2, 'dt', self.dt)
		self.orig_df = self.df
		
		# Extract feature
		self.df = self.df[feature]
		# Create an array of sequences
		self.data = [torch.from_numpy(np.array(self.df[i : i + self.seq_len])) 
			for i in range(self.df.size - self.seq_len)]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx: int):
		return self.data[idx]

	def visualize(self):
		self.orig_df.plot(x = 'DATE', y = 'Value', kind = 'line')
		pyplot.show()

	def sample(self):
		pass