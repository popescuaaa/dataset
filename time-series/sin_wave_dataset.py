import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as pyplot
import pandas as pd
import numpy as np

class SinWave(Dataset):
	def __init__(self, csv_path: str,  seq_len: int):
		self.seq_len = seq_len
		self.df = pd.read_csv(csv_path)

		# Compte ∆t (deltas)
		self.dt = np.array([(self.df.Wave[i+1] - self.df.Wave[i]) 
			for i in range(self.df.Wave.size - 1)])
		self.dt = np.concatenate([np.array([0]), self.dt])

		# Append ∆t (deltas)
		self.df.insert(1, 'dt', self.dt)

		# Create two structures for data and ∆t
		self.sinewave_data = [torch.from_numpy(np.array(self.df.Wave[i : i + self.seq_len])) 
			for i in range(self.df.size - self.seq_len)]
		
		self.dt_data = [torch.from_numpy(np.array(self.df.dt[i : i + self.seq_len])) 
			for i in range(self.df.size - self.seq_len)]

	def __len__(self):
		return len(self.sinewave_data)

	def __getitem__(self, idx: int):
		return self.sinewave_data[idx], self.dt_data[idx]
		

if __name__ == '__main__':
	sinwave_csv_path = './csv/sinwave.csv'
	ds = SinWave(sinwave_csv_path, 100)
	dl = DataLoader(ds, batch_size=10, shuffle=False, num_workers=10) # explicit no shuffle!!
	
