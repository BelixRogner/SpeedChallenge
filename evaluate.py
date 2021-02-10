import matplotlib.pyplot as plt
from FrameDataset import FrameDataset
import torch
import time
from utils import loadFile, loadLabels, loadStats
import tqdm
import numpy as np
import os

def moving_average(x, w):
	return np.convolve(x, np.ones(w), 'valid') / w

def calcAveraged(avg, data):
	temp = moving_average(data, avg)
	start = np.full(avg//2, temp[0])
	end = np.full(avg//2+avg%2-1, temp[-1])
	return np.concatenate((start, temp, end))

HEIGHT = 100
WIDTH = 640
files = [file for file in os.listdir("results") if file.endswith(".pt")]
print(len(files))
MODELS = files

BATCH_SIZE = 16

SEQ_LEN = 9

offset = SEQ_LEN // 2
print(SEQ_LEN, BATCH_SIZE)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

frames = loadFile("data/test.mp4", HEIGHT, WIDTH)
stats = loadStats("data/train.mp4_modified_100_640_96.pt")

pb = tqdm.tqdm()

test = FrameDataset(frames, stats, torch.zeros(len(frames)), train=False, seqLen=SEQ_LEN, numSplits=1, splitIdxs=torch.tensor([0]))
testLoader = torch.utils.data.DataLoader(test, batch_size=BATCH_SIZE, shuffle=False, num_workers=6, prefetch_factor=1)
speeds = []

for model in MODELS:
	speed = []
	pb.reset(len(testLoader))
	pb.set_description(f"Evaluating...")
	net = torch.load(f"results/{model}")
	net.to(device)
	net.eval()
	with torch.no_grad():
		for data in testLoader:
			inputs = data[0].to(device)
			speed.append(torch.nn.functional.relu(net(inputs), inplace=True))
			pb.update()
	speed = torch.cat(speed, dim=0).view(-1).cpu()
	speed = np.concatenate((np.full(offset, speed[0]), speed, np.full(offset, speed[-1])))
	assert len(frames) == len(speed)
	speed[1130:1618] = 0.0
	speeds.append(speed)

while True:
	MOVING_AVG = int(input("average: "))
	for i in range(len(speeds)):
		plt.plot(calcAveraged(MOVING_AVG, speeds[i]), label=MODELS[i])
	
	average = calcAveraged(MOVING_AVG, np.mean(speeds, axis=0))
	plt.plot(average, label="Average")

	#plt.plot(calcAveraged(MOVING_AVG, np.median(speeds, axis=0)), label="Median")
	
	losses = np.zeros(len(speeds))
	for i in range(len(speeds)):
		for j in range(i+1, len(speeds)):
			loss = np.mean((calcAveraged(MOVING_AVG, speeds[i])-calcAveraged(MOVING_AVG, speeds[j]))**2)
			print(i,j,loss)
			losses[i] += loss
			losses[j] += loss

	print(losses/(len(speeds)-1))
	plt.legend()
	plt.show()
	with open(f'test_{MOVING_AVG}.txt', 'w') as f:	
		for x in average:
			f.write(f"{x.item()}\n")

	